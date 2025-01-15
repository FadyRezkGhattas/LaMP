import os
import yaml
import json
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from functools import partial

import torch
import numpy as np
import nevergrad as ng
import torch.nn.functional as F
from transformers.trainer_callback import PrinterCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.lmdb import LMDBDataset
from torch.utils.data import Subset
from metrics.utils import get_metrics
from lora_xs.make_peft_model import make_peft_model
from prompts.singular_prompts import create_prompt_generator
from load_adapters import tensorize_loraxs_adapter
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="", help="used to log results in ./experiments/{task}/{dataset_name}_lora_hub")
parser.add_argument("--log_prefix", default="", help="used to log results in ./experiments/{task}/{dataset_name}_lora_hub")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_5/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-5/finetune_all_train_user_profiles/checkpoint-144000')
parser.add_argument("--svd_pth", default='./experiments/fixed_adapter')
parser.add_argument("--use_bf16", default=True)
parser.add_argument("--from_user_id", type=int, default=0, help="Train model starting from this user index.")
parser.add_argument("--to_user_id", type=int, default=-1, help="Terminate training at this user index. If -1, train until end of available users.")
parser.add_argument("--task", default='LaMP-5')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--num_tasks', type=int, default=-1, help='total number of tasks to evaluate model zoo on. If -1, all users are evaluated.')
parser.add_argument('--early_stop', type=int, default=1e10, help='how many steps to wait for performance to not improve before skipping the rest of the model zoo')
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')
parser.add_argument('--profile_training_ratio', type=float, default=None, help='A ratio to split the profile into training and validation sets. The split ratio If None, no split will be performed.')
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-5-final')

def make_linear_combinations_adapter(selected_adapters, weights):
    adapter = {}
    # all keys are the same
    keys = selected_adapters[0].keys()
    for i, selected_adapter in enumerate(selected_adapters):
        if i == 0:
            for key in keys:
                adapter[key] = weights[i] * selected_adapter[key]
        else:
            for key in keys:
                adapter[key] = (
                    adapter[key] + weights[i] * selected_adapter[key]
                )
    return adapter

def get_support_metrics(opts, model, tokenizer, collator, final_adapter, profile_data, compute_metrics, best_metric):
    _ = model.load_state_dict(final_adapter, strict=False)

    support_loss_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        eval_accumulation_steps = 1,
        disable_tqdm=True,
        bf16=opts.use_bf16,
        # generation args
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        generation_max_length = opts.max_generation_length,
    )
    loss_evaluator = Seq2SeqTrainer(
        model = model,
        args = support_loss_args,
        data_collator = collator,
        eval_dataset = profile_data,
        tokenizer = tokenizer,
        compute_metrics=compute_metrics
    )
    loss_evaluator.remove_callback(PrinterCallback)

    # insert adapter into model
    results = loss_evaluator.evaluate(profile_data)
    
    return results[f"eval_{best_metric}"]

if __name__ == "__main__":
    opts = parser.parse_args()

    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    folder_name = f'{dataset_name}_lora_hub'
    folder_name = folder_name + f'_{opts.exp_name}' if opts.exp_name != "" else folder_name
    output_dir = os.path.join('./experiments', opts.task, folder_name)
    log_files_pth = os.path.join(output_dir, 'per_user')

    # Log hyperparameters
    os.makedirs(output_dir, exist_ok=True)

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
    print("Loading Adapters")
    model_zoo = LMDBDataset(opts.lmdb_addr)

    # tensorize model zoo
    print("Tensorizing finite hypothesis")
    adapters = []
    for i in range(len(model_zoo)):
        adapters.append(tensorize_loraxs_adapter(model_zoo[i], use_bf16=opts.use_bf16))

    prompt_generator = create_prompt_generator(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)

    from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)
    for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
        user_log = os.path.join(log_files_pth, f'{opts.log_prefix}results_user_{user_id}.json')
        with open(user_log) as f:
            user_log_data = json.load(f)

        # load user profile and query
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size, training_ratio=opts.profile_training_ratio)
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

        # get the chosen N adapter and weights
        chosen_adapters_ids = user_log_data['chosen_adapters']
        weights = user_log_data['weights']
        selected_adapters = [adapters[adapter_id] for adapter_id in chosen_adapters_ids]

        # construct final state dict, load to model and make prediction
        final_adapter = make_linear_combinations_adapter(selected_adapters, weights)

        # get support metrics
        best_metric_support_value = get_support_metrics(opts, original_model, tokenizer, collator, final_adapter, profile_data, compute_metrics, best_metric)
        user_log_data['best_support_metric'] = best_metric_support_value

        if not os.path.exists(log_files_pth):
            os.makedirs(log_files_pth)
        # os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json'
        with open(user_log, 'w') as file:
            json.dump(user_log_data, file, indent = 4)