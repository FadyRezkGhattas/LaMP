import os
import yaml
import json
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers.trainer_callback import PrinterCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.lmdb import LMDBDataset
from metrics.utils import get_metrics
from lora_xs.make_peft_model import make_peft_model
from prompts.singular_prompts import create_prompt_generator
from load_adapters import tensorize_loraxs_adapter
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset

from diffusion.net import get_model
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.sample import posterior_sample

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="diff", help="used to log results in ./experiments/{task}/{dataset_name}_stage_4_{exp_name}")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--svd_pth", default='./experiments/fixed_adapter')
parser.add_argument("--use_bf16", default=True)
parser.add_argument("--from_user_id", type=int, default=0, help="Train model starting from this user index.")
parser.add_argument("--to_user_id", type=int, default=-1, help="Terminate training at this user index. If -1, train until end of available users.")
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--truncate_profile_size', type=int, default=64, help='if > 0, then the profile size is truncated to max of given value.')

# diffusion model and model zoo
parser.add_argument('--reverse_z_score', type=bool, default=True, help='If True, the lmdb dataset statistics are computed to reverse z-score of model')
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-2-final')
parser.add_argument('--truncate_lmdb_dataset', type=int, default=-1, help='if > 0, then the lmdb dataset will be subsampled to have len=truncate_lmdb_dataset.')
parser.add_argument('--diff_ckpt', type=str, default='./experiments/LaMP-2/diffusion/LaMP-2_normalize_data_3x_241007_204226/final_ckpt.pt', help='path to diffusion model for sampling model zoo')
parser.add_argument('--diff_hdim', type=int, default=7680, help='hidden dim of diff net')
parser.add_argument('--diff_nhids', type=int, default=3, help='num of hidden layers in diff net')
parser.add_argument('--diff_odim', type=int, default=2592, help='size of input and output dimensionality of the diffusion model')

def collect_grads(model):
    """Collect gradients from all parameters"""
    grads = []
    for param in model.parameters():
        if param.requires_grad:
            grads.append(param.grad)
    return torch.stack(grads, dim=0)

def get_loss_grads(model, adapter, batch, mean, std):
    model.train()
    device = adapter.device
    adapter = mean.to(device) + (adapter*std.to(device))
    adapter = tensorize_loraxs_adapter(adapter)
    _ = original_model.load_state_dict(adapter, strict=False)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(**batch)[0]
    loss.backward()
    grads = collect_grads(model)
    model.eval()
    return loss.item(), grads

def eval_adapters_losses_user(opts, original_model, collator, profile_data, gaussian_diff, diffusion_net, get_loss_grads_):
    original_model = original_model.to(dtype=torch.bfloat16, device='cuda')
    loader = DataLoader(profile_data, opts.per_device_batch_size, shuffle=False, collate_fn=collator, drop_last=False, pin_memory=True)
    i, batch = next(enumerate(loader))
    adapters, losses = posterior_sample(gaussian_diff, diffusion_net, original_model, get_loss_grads_, batch, opts.cand_size, opts.timestep_dps)
    return adapters, losses

def eval_adapters_accuracies_user(opts, output_dir, original_model, collator, tokenizer, compute_metrics, adapters, best_adapters_idx, profile_data):
    support_acc_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        generation_num_beams = 1,
        predict_with_generate = True,
        eval_accumulation_steps = 1,
        generation_max_length = opts.max_generation_length,
        disable_tqdm=True,
        bf16=opts.use_bf16
    )
    acc_evaluator = Seq2SeqTrainer(
        model = original_model,
        args = support_acc_args,
        data_collator = collator,
        eval_dataset = profile_data,
        tokenizer = tokenizer,
        compute_metrics=compute_metrics
    )
    acc_evaluator.remove_callback(PrinterCallback)

    # evaluate accuracy on these best k adapters
    best_adapters_accuracies = []
    for adapter_id in tqdm(best_adapters_idx, leave=False, desc=f'Top Adapters', position=1):
        # insert adapter into model
        _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
        
        results = acc_evaluator.evaluate(profile_data)
        adapter_selection_metric_val = results['eval_'+best_metric]
        best_adapters_accuracies.append(adapter_selection_metric_val)
    return best_adapters_accuracies

def get_adapter_prediction(opts, original_model, tokenizer, adapters, adapter_id, query_data):
    _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
    # get query prediction
    inputs = tokenizer(query_data[0]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
    outputs = original_model.generate(**inputs, num_beams=opts.generation_num_beams, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
    outputs = outputs.to('cpu')
    tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))
    return tokenized_prediction

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
    original_model = make_peft_model(opts, original_model)

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

    with open(opts.data_addr) as f:
        user_data = json.load(f)

    # Loading model zoo
    print("Loading/Sampling Adapters")
    # load diffusion model
    diffusion_net = get_model(opts).to('cuda')
    # load diffusion sampler
    gaussian_diff = GaussianDiffusion().to('cuda')
    
    if opts.reverse_z_score:
        mean, std = LMDBDataset(opts.lmdb_addr).get_data_stats()
    else:
        mean, std = 0, 1
    get_loss_grads_ = partial(get_loss_grads, mean=mean, std=std)

    prompt_generator = create_prompt_generator(tokenizer)
    
    user_ids = [] # shape: number_users x 1
    tokenized_predictions = [] # shape: number_users x 1
    txt_labels = [] # shape: number_users x 1
    best_adapter_ids = [] # shape: number_users x 1
    support_performance_all_users = [] # shape: number_users x num_adapters performance of adapters on user profiles
    best_train_metrics = [] # shape: number_users x1
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)

    from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)
    for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
        if Path(os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json')).is_file():
            continue
        user_ids.append(user_id)

        # load user profile and query
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size)
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
        query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, user_id=user_id, data=user_data[user_id])
        txt_labels.append(query_data[0]['target'])
        query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

        t0 = time.time()
        # Get losses of all adapters
        adapters, user_support_perf = eval_adapters_losses_user(opts, original_model, collator, profile_data, gaussian_diff, diffusion_net, get_loss_grads_)
        # Get best 15 adapters indices
        best_15_adapters_idx = np.argsort(user_support_perf)[:15]
        # get accuracies on best 15 adapters. predictions are generated with greedy sampling
        best_15_adapters_accuracies = eval_adapters_accuracies_user()
        # Get beam searched prediction on best adapter of the shortlisted 15
        best_adapter_id = best_15_adapters_idx[np.argmax(best_15_adapters_accuracies)]
        tokenized_prediction = get_adapter_prediction()
        t1 = time.time()

        tokenized_predictions.append(tokenized_prediction)
        best_adapter_ids.append(int(best_adapter_id))
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
                'best_adapter_ids': int(best_adapter_id),
                'user_train_perfs': user_support_perf,
                'best_15_adapters_accuracies': best_15_adapters_accuracies,
                'adapters_eval_time': t1-t0
            }, file, indent = 4)
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