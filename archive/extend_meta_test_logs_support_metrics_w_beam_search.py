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
from peft import LoraConfig, get_peft_model
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

# from diffusion.net import get_model
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.sample import greedy_sample

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="mlp4x3l_5_clusters_zoo", help="used to log results in ./experiments/{task}/{dataset_name}_stage_4_{exp_name}")
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
parser.add_argument('--num_tasks', type=int, default=-1, help='total number of tasks to evaluate model zoo on. If -1, all users are evaluated.')
parser.add_argument('--early_stop', type=int, default=1e10, help='how many steps to wait for performance to not improve before skipping the rest of the model zoo')
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')

# diffusion model and model zoo
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-2-final-4x3l-diff-samples')
parser.add_argument('--lmdb_clusters', type=str, default=None, help='If provided, the medoids are used to chose a cluster whose adapters are evaluated for a user.')

def eval_adapters_accuracies_user(opts, output_dir, original_model, collator, tokenizer, compute_metrics, adapters, best_adapters_idx, profile_data, generation_num_beams):
    support_acc_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        generation_num_beams = generation_num_beams,
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

if __name__ == '__main__':
    opts = parser.parse_args()
    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./experiments', opts.task, f'{dataset_name}_stage_4_{opts.exp_name}')
    log_files_pth = os.path.join(output_dir, 'per_user')

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "hyperparameters.json"), 'w') as f:
        json.dump(vars(opts), f)

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
    
    user_ids = [] # shape: number_users x 1
    tokenized_predictions = [] # shape: number_users x 1
    txt_labels = [] # shape: number_users x 1
    best_adapter_ids = [] # shape: number_users x 1
    support_performance_all_users = [] # shape: number_users x num_adapters performance of adapters on user profiles
    best_train_metrics = [] # shape: number_users x1
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)

    eval_adapters_accuracies_user_ = partial(eval_adapters_accuracies_user, opts=opts, output_dir=output_dir, original_model=original_model, collator=collator, tokenizer=tokenizer, compute_metrics=compute_metrics, adapters=adapters)

    for user_id in tqdm(range(len(user_data)), leave=True, desc='Users', position=0):
        user_log = os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json')
        with open(user_log) as f:
            user_log_data = json.load(f)
        best_adapter_id = user_log_data['best_adapter_ids']

        # load user profile and query
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size)
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

        best_adapter_support_metrics = eval_adapters_accuracies_user_(best_adapters_idx=[best_adapter_id], profile_data=profile_data, generation_num_beams=opts.generation_num_beams)

        user_log_data['best_adapter_support_metrics'] = best_adapter_support_metrics

        with open(user_log, 'w') as file:
            json.dump(user_log_data, file, indent = 4)