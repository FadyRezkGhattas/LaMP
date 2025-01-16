import os
import sys
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import yaml
import json
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer)

from data.lmdb import LMDBDataset
from metrics.utils import get_metrics
from load_adapters import load_adapter, tensorize_loraxs_adapter
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset
from lora_xs.initialization_utils import find_and_initialize
from prompts.singular_prompts import create_prompt_generator


parser = argparse.ArgumentParser()
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_3/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-3/finetune_all_train_user_profiles/checkpoint-117500')
parser.add_argument("--task", default='LaMP-3')
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument('--truncate_profile_size', type=int, default=256, help='if > 0, then the profile size is truncated to max of given value.')

# This can be mezo/sgd/zoo/diff/lorahub log files root path.
# mezo/sgd will have ckpts folder with ckpts for every user (ckpts)
# and zoo/diff/lorahub would have a per_user folder with a json for every user (zoo selection)
parser.add_argument('--results_addr', type=str, default='experiments/LaMP-3/sgd_baseline/r_6_alpha_16_lr_0.01_epochs_20_sch_linear/')
parser.add_argument("--algorithm", type=str, choices=['zoo_selection', 'lora_hub', 'ckpts'])
# the lmdb data used if algorithm is zoo_selection
parser.add_argument("--lmdb_addr", type=str, default='lmdb_data/LaMP-3-final')

def get_adapter_metrics(opts, user_model, tokenizer, data, compute_metrics):
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)
    training_args = Seq2SeqTrainingArguments(
            # trainer basics
            output_dir=opts.results_addr,
            do_train = True,
            do_eval = True,
            # parallelization args
            per_device_eval_batch_size = opts.per_device_batch_size,
            # generation args
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate = True,
            generation_max_length = opts.generation_max_length,
        )
    trainer = Seq2SeqTrainer(
        model = user_model,
        args = training_args,
        data_collator = collator,
        eval_dataset = data,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    return trainer.evaluate(metric_key_prefix='support')

def get_user_json_results(opts, user_id):
    user_log = f'user_{user_id}.json'
    results_path = os.path.join(opts.results_addr, 'per_user')
    users = os.listdir(results_path)
    users = [user for user in users if user_log in user]
    assert len(users) == 1
    user_file_path = users[0]
    user_file_path = os.path.join(results_path, user_file_path)
    with open(user_file_path, 'r') as f:
        user_log = json.load(f)
    return user_log

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

def load_adapter_based_on_method(user_id, original_model, model_zoo, opts):
    # if zoo selection, then get the best adapter id and load it
    if opts.algorithm in 'zoo_selection' :
        user_log = get_user_json_results(opts, user_id)
        user_adapter_id = user_log['best_adapter_ids']
        _ = original_model.load_state_dict(model_zoo[user_adapter_id], strict=False)
        return original_model
    
    # if lora hub, then get selected adapters and corresponding weights to create final adapter and load it
    elif opts.algorithm == 'lora_hub':
        user_log = get_user_json_results(opts, user_id)
        chosen_adapters_ids = user_log['chosen_adapters']
        weights = user_log['weights']
        selected_adapters = [model_zoo[adapter_id] for adapter_id in chosen_adapters_ids]
        final_adapter = make_linear_combinations_adapter(selected_adapters, weights)
        _ = original_model.load_state_dict(final_adapter, strict=False)
        return original_model

    # if ckpts (mezo or sgd), then load the corresponding checkpoint for the user
    elif opts.algorithm == 'ckpts':
        user_model = load_adapter(original_model, os.path.join(opts.results_addr, 'ckpts', f'user_{user_id}')).to('cuda')
        return user_model
    
    else:
        raise ValueError(f'Got {opts.algorithm} but should be in [zoo_selection, lora_hub, ckpts]')

if __name__ == '__main__':
    opts = parser.parse_args()
    log_files_pth = os.path.join(opts.results_addr, 'per_user_cross_entropy')
    with open(os.path.join(opts.results_addr, "cross_entropy_hyperparameters.json"), 'w') as f:
        json.dump(vars(opts), f)

    print("Loading Model")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="./cache")
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
        writer=None, reconstruct_config=reconstr_config, skip_svd=True
        )
    original_model.print_trainable_parameters()

    print("loading Dataset")
    # Load all users data    
    task = opts.task
    compute_metrics, best_metric, txt_labels, greater_is_better = get_metrics(task, tokenizer)

    prompt_generator = create_prompt_generator(tokenizer)

    with open(opts.data_addr) as f:
        user_data = json.load(f)

    if opts.algorithm in ['zoo_selection', 'lora_hub']:
        print("Loading Adapters")
        lmdb_data = LMDBDataset(opts.lmdb_addr)
        print("Tensorizing finite hypothesis")
        model_zoo = []
        for i in range(len(lmdb_data)):
            model_zoo.append(tensorize_loraxs_adapter(lmdb_data[i], use_bf16=True))
    elif opts.algorithm == 'ckpts':
        model_zoo = None
        
    tokenized_predictions = []
    txt_predictions = []
    txt_labels = []
    user_ids = []
    num_users = len(user_data)
    with tqdm(total=num_users, desc='Processing Users') as pbar:
        for user_id in range(num_users):
            user_ids.append(user_id)
            # figure out the method, and use different methods
            user_model = load_adapter_based_on_method(user_id, original_model, model_zoo, opts)

            # Get support performance
            data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size, training_ratio=opts.profile_training_ratio)
            data = convert_to_hf_dataset(data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
            support_results = get_adapter_metrics(opts, user_model, tokenizer, data, compute_metrics)

            # Get query performance
            data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size, training_ratio=opts.profile_training_ratio)
            
            if not os.path.exists(log_files_pth):
                os.makedirs(log_files_pth)
            # with open(os.path.join(log_files_pth, f'user_{user_id}.json'), 'w') as file:
            #     json.dump({}, file, indent = 4)

            pbar.update(1)