import os
import yaml
import json
import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers.trainer_callback import PrinterCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.lmdb import LMDBDataset
from metrics.utils import get_metrics
from load_adapters import tensorize_loraxs_adapter
from lora_xs.initialization_utils import find_and_initialize
from prompts.singular_prompts import create_prompt_generator
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset

from diffusion.net import get_model
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.sample import greedy_sample

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="mix_acc_loss_selection_beam1", help="used to log results in ./experiments/{task}/{dataset_name}_stage_4_{exp_name}")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--use_bf16", default=True)
parser.add_argument("--from_user_id", type=int, default=0, help="Train model starting from this user index.")
parser.add_argument("--to_user_id", type=int, default=-1, help="Terminate training at this user index. If -1, train until end of available users.")
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--num_tasks', type=int, default=-1, help='total number of tasks to evaluate model zoo on. If -1, all users are evaluated.')
parser.add_argument('--early_stop', type=int, default=1e10, help='how many steps to wait for performance to not improve before skipping the rest of the model zoo')
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')
parser.add_argument('--selection_metric', type=str, choices=['loss', 'best_metric'], default='best_metric', help='Whether to use support loss for adapter selection or dataset specific metric (best_metric)')
parser.add_argument('--track_query', default=False, help='Whether to calculate query for every adapter. Computationally expensive for selection_metric=`best_metric`')

# diffusion model and model zoo
parser.add_argument('--use_diffusion', type=bool, default=False)
parser.add_argument('--reverse_z_score', type=bool, default=True, help='If True, the lmdb dataset statistics are computed to reverse z-score of model')
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-2-final')
parser.add_argument('--diff_ckpt', type=str, default='./experiments/LaMP-2/diffusion/LaMP-2_normalize_data_3x_241007_204226/final_ckpt.pt', help='path to diffusion model for sampling model zoo')
parser.add_argument('--diff_hdim', type=int, default=7680, help='hidden dim of diff net')
parser.add_argument('--diff_nhids', type=int, default=3, help='num of hidden layers in diff net')
parser.add_argument('--diff_odim', type=int, default=2592, help='size of input and output dimensionality of the diffusion model')

if __name__ == '__main__':
    opts = parser.parse_args()
    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./experiments', opts.task, f'{dataset_name}_stage_4_{opts.exp_name}')
    log_files_pth = os.path.join(output_dir, 'per_user')

    print("Loading Model")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="./cache")
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    generation_config = GenerationConfig.from_pretrained(opts.model_name)

    # prepare model for PEFTing
    print("Configuring LoRA-XS")
    rank = 6
    lora_alpha = 16
    config = LoraConfig(
        r=rank,
        target_modules=["q", "v"],
        task_type="SEQ_2_SEQ_LM", # assuming a decoder-only model in this example
        lora_alpha=lora_alpha,
        use_rslora=True
        )
    original_model = get_peft_model(original_model, config)
    with open("./lora_xs/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    adapter_name = "default"  # assuming a single LoRA adapter per module should be transformed to LoRA-XS
    peft_config_dict = {adapter_name: config}
    reconstr_config['svd']['rank'] = rank
    find_and_initialize(
        original_model, peft_config_dict, adapter_name=adapter_name, reconstr_type='svd',
        writer=None, reconstruct_config=reconstr_config
        )
    original_model.print_trainable_parameters()
    original_model = original_model.to('cuda')
    original_model.eval()
    for name, param in original_model.named_parameters():
        param.contiguous()
    # converting to bf16 after initializing lora-xs because sklearn svd does not support bf16 dtype
    if opts.use_bf16:
        original_model = original_model.bfloat16()

    # Load all users data
    print("Loading Dataset")
    task = opts.task
    compute_metrics, best_metric, txt_labels, greater_is_better = get_metrics(task, tokenizer)
    predict_with_generate = True
    if opts.selection_metric == 'loss':
        compute_metrics, greater_is_better, best_metric, predict_with_generate = None, False, 'loss', False

    with open(opts.data_addr) as f:
        user_data = json.load(f)

    # Loading model zoo
    print("Loading/Sampling Adapters")
    if opts.use_diffusion:
        # load diffusion model
        diffusion_net = get_model(opts).to('cuda')
        # load diffusion sampler
        gaussian_diff = GaussianDiffusion().to('cuda')
        # sample model zoo
        model_zoo = greedy_sample(gaussian_diff, diffusion_net)
        # reverse batch z-score if necessary
        if opts.reverse_z_score:
            mean, std = LMDBDataset(opts.lmdb_addr).get_data_stats()
            device = model_zoo[0].device
            model_zoo = [mean.to(device) + (x*std.to(device)) for x in model_zoo]
    else:
        model_zoo = LMDBDataset(opts.lmdb_addr)
    # tensorize model zoo
    print("Tensorizing finite hypothesis")
    adapters = []
    for i in range(len(model_zoo)):
        adapters.append(tensorize_loraxs_adapter(model_zoo[i], use_bf16=opts.use_bf16))

    prompt_generator = create_prompt_generator(tokenizer)
    
    user_ids = [] # shape: number_users x 1
    tokenized_predictions = [] # shape: number_users x 1
    txt_labels = [] # shape: number_users x 1
    best_adapter_ids = [] # shape: number_users x 1
    support_performance_all_users = [] # shape: number_users x num_adapters performance of adapters on user profiles
    best_train_metrics = [] # shape: number_users x1
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)
    task_counter = 0
    from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)

    user_results_dir = 'experiments/LaMP-2/dev_questions_merged_stage_4_full_model_zoo_stage5/per_user/'
    for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
        if Path(os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json')).is_file():
            continue
        user_ids.append(user_id)
        user_support_perf = [] # shape: num_adapters
        user_query_perf = [] # shape: num_adapters x1

        # load user profile and query
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size)
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
        query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, user_id=user_id, data=user_data[user_id])
        txt_labels.append(query_data[0]['target'])
        query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            do_eval = True,
            per_device_eval_batch_size = opts.per_device_batch_size,
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate = predict_with_generate,
            eval_accumulation_steps = 1,
            generation_max_length = opts.max_generation_length,
            disable_tqdm=True,
            bf16=opts.use_bf16
        )
        trainer = Seq2SeqTrainer(
            model = original_model,
            args = training_args,
            data_collator = collator,
            eval_dataset = profile_data,
            tokenizer = tokenizer,
            compute_metrics = compute_metrics
        )
        trainer.remove_callback(PrinterCallback)

        # find best k adapters with min support loss
        with open(os.path.join(user_results_dir, f'full_model_zoo_stage5results_user_{user_id}.json')) as f:
            user_support_losses = json.load(f)['user_train_perfs']
        # evaluate accuracy on these best k adapters
        indices = np.argsort(user_support_losses)[:15]
        for adapter_id in tqdm(indices, leave=False, desc='Adapters', position=1):
            # insert adapter into model
            _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
            
            results = trainer.evaluate(profile_data)
            adapter_selection_metric_val = results['eval_'+best_metric]
            user_support_perf.append(adapter_selection_metric_val)
        
        # find best adapter using max support acc and load
        best_adapter_id = indices[np.argmax(user_support_perf)]
        _ = original_model.load_state_dict(adapters[best_adapter_id], strict=False)
        # get query prediction
        inputs = tokenizer(query_data[0]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
        outputs = original_model.generate(**inputs, num_beams=opts.generation_num_beams, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
        outputs = outputs.to('cpu')
        tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))

        tokenized_predictions.append(tokenized_prediction)
        best_adapter_ids.append(best_adapter_id)
        support_performance_all_users.append(user_support_perf)

        # log user final results
        txt_prediction = tokenizer.decode(tokenized_prediction, skip_special_tokens=True)
        if not os.path.exists(log_files_pth):
            os.makedirs(log_files_pth)
        with open(os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json'), 'w') as file:
            json.dump({
                'user_ids': user_id,
                'label': txt_labels[-1],
                'pred': txt_prediction,
                'user_train_perfs': user_support_perf,
                'query_losses': user_query_perf,
            }, file, indent = 4)
        task_counter += 1
        if task_counter == opts.num_tasks:
            break
    txt_predictions = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)
    
    tokenized_labels = tokenizer(txt_labels)['input_ids']
    tokenized_labels = np.array([np.pad(torch.tensor(x), (tokenizer.pad_token_id, opts.max_generation_length - len(x))) for x in tokenized_labels])
    results = compute_metrics((tokenized_predictions, tokenized_labels))
    print(results)
    
    with open(os.path.join(output_dir, f'{opts.exp_name}results.json'), 'w') as file:
        json.dump({
            **results,
            'user_ids':user_ids,
            'labels':txt_labels,
            'preds': txt_predictions,
            'best_adapter_ids': best_adapter_ids,
            'user_train_perfs': support_performance_all_users,
            'best_train_metrics': best_train_metrics
        }, file, indent = 4)