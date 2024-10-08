import os
import yaml
import json
import argparse
from tqdm import tqdm

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

parser = argparse.ArgumentParser()
parser.add_argument("--exp_prefix", default="")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--lmdb_addr', type=str, default=None)
parser.add_argument('--num_tasks', type=int, default=-1, help='total number of tasks to evaluate model zoo on. If -1, all users are evaluated.')
parser.add_argument('--early_stop', type=int, default=1e10, help='how many steps to wait for performance to not improve before skipping the rest of the model zoo')
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')

# diffusion model 
parser.add_argument('--diff_ckpt', type=str, default='./experiments/LaMP-2/diffusion/LaMP-2_normalize_data_3x_241007_204226/final_ckpt.pt', help='path to diffusion model for sampling model zoo')
parser.add_argument('--diff_hdim', type=int, default=7680, help='hidden dim of diff net')
parser.add_argument('--diff_nhids', type=int, default=3, help='num of hidden layers in diff net')
parser.add_argument('--diff_odim', type=int, default=2592, help='size of input and output dimensionality of the diffusion model')

if __name__ == '__main__':
    opts = parser.parse_args()
    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./experiments', opts.task, f'{dataset_name}_model_zoo_as_finite_hypothesis')
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

    print("loading Dataset")
    # Load all users data    
    task = opts.task
    compute_metrics, best_metric, txt_labels, greater_is_better = get_metrics(task, tokenizer)
    with open(opts.data_addr) as f:
        user_data = json.load(f)

    # Loading model zoo
    if opts.lmdb_addr is not None:
        model_zoo = LMDBDataset(opts.lmdb_addr)
    elif opts.diff_ckpt is not None:
        # load diffusion model
        diffusion_net = get_model(opts).to('cuda')
        # load diffusion sampler
        gaussian_diff = GaussianDiffusion().to('cuda')
        # sample model zoo
        model_zoo = greedy_sample(gaussian_diff, diffusion_net)
    # tensorize model zoo
    print("Tensorizing finite hypothesis")
    adapters = []
    for i in range(len(model_zoo)):
        adapters.append(tensorize_loraxs_adapter(model_zoo[i]))

    prompt_generator = create_prompt_generator(tokenizer)
    
    user_ids = [] # shape: number_users x 1
    tokenized_predictions = [] # shape: number_users x 1
    txt_labels = [] # shape: number_users x 1
    best_adapter_ids = [] # shape: number_users x 1
    user_train_perfs = [] # shape: number_users x num_adapters performance of adapters on user profiles
    best_train_metrics = [] # shape: number_users x1
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)
    num_tasks = opts.num_tasks if opts.num_tasks != -1 else len(user_data)
    for user_id in tqdm(range(num_tasks), desc='User', position=0):
        user_ids.append(user_id)
        user_train_perf = [] # shape: num_adapters
        # load user profile and query
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size)
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
        query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, user_id=user_id, data=user_data[user_id])
        txt_labels.append(query_data[0]['target'])

        training_args = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            do_eval = True,
            per_device_eval_batch_size = opts.per_device_batch_size,
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate = True,
            eval_accumulation_steps = 1,
            generation_max_length = opts.max_generation_length,
            disable_tqdm=True
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

        for adapter_id in tqdm(range(len(adapters)), leave=False, desc='Adapters', position=1):
            # insert adapter into model
            _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
            
            results = trainer.evaluate(profile_data)
            adapter_selection_metric_val = results['eval_'+best_metric]
            
            if greater_is_better:
                best_flag = (len(user_train_perf)==0) or (adapter_selection_metric_val > np.max(user_train_perf))
            else:
                best_flag = (len(user_train_perf)==0) or (adapter_selection_metric_val < np.min(user_train_perf))

            if best_flag:
                best_train_metric = adapter_selection_metric_val
                best_adapter_id = adapter_id
                inputs = tokenizer(query_data[0]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
                outputs = original_model.generate(**inputs, num_beams=4, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
                outputs = outputs.to('cpu')
                tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))

            user_train_perf.append(adapter_selection_metric_val)
            
            if len(user_train_perf)>opts.early_stop:
                early_stop = all(x <= best_train_metric for x in user_train_perf[-opts.early_stop:]) if greater_is_better else all(x >= best_train_metric for x in user_train_perf[-opts.early_stop:])
                if early_stop:
                    break
        
        tokenized_predictions.append(tokenized_prediction)
        best_adapter_ids.append(best_adapter_id)
        user_train_perfs.append(user_train_perf)
        best_train_metrics.append(best_train_metric)

        # log user final results
        txt_prediction = tokenizer.decode(tokenized_prediction, skip_special_tokens=True)
        if not os.path.exists(log_files_pth):
            os.makedirs(log_files_pth)
        with open(os.path.join(log_files_pth, f'{opts.exp_prefix}results_user_{user_id}.json'), 'w') as file:
            json.dump({
                'user_ids': user_id,
                'label': txt_labels[-1],
                'pred': txt_prediction,
                'best_adapter_ids': best_adapter_id,
                'user_train_perfs': user_train_perf,
                'best_train_metric': best_train_metric
            }, file, indent = 4)

    txt_predictions = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)
    
    tokenized_labels = tokenizer(txt_labels)['input_ids']
    tokenized_labels = np.array([np.pad(torch.tensor(x), (tokenizer.pad_token_id, opts.max_generation_length - len(x))) for x in tokenized_labels])
    results = compute_metrics((tokenized_predictions, tokenized_labels))
    print(results)
    
    with open(os.path.join(output_dir, f'{opts.exp_prefix}results.json'), 'w') as file:
        json.dump({
            **results,
            'user_ids':user_ids,
            'labels':txt_labels,
            'preds': txt_predictions,
            'best_adapter_ids': best_adapter_ids,
            'user_train_perfs': user_train_perfs,
            'best_train_metrics': best_train_metrics
        }, file, indent = 4)