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
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--exp_name", default="", help="used to log results in ./experiments/{task}/{dataset_name}_lora_hub")
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
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-2-final')
parser.add_argument('--lmdb_clusters', type=str, default=None, help='If provided, the medoids are used to chose a cluster whose adapters are evaluated for a user.')

# LoraHub hyperparameters
parser.add_argument('--num_lorahub_adapters', type=int, default=20)
parser.add_argument('--max_inference_step', type=int, default=40)

def get_adapter_loss(opts, output_dir, model, collator, tokenizer, profile_data):    
    support_loss_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        eval_accumulation_steps = 1,
        generation_max_length = opts.max_generation_length,
        disable_tqdm=True,
        bf16=opts.use_bf16
    )
    loss_evaluator = Seq2SeqTrainer(
        model = model,
        args = support_loss_args,
        data_collator = collator,
        eval_dataset = profile_data,
        tokenizer = tokenizer
    )
    loss_evaluator.remove_callback(PrinterCallback)

    # insert adapter into model
    results = loss_evaluator.evaluate(profile_data)
    adapter_selection_metric_val = results['eval_loss']
    
    return adapter_selection_metric_val

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

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

def get_score(weights, model, profile_data, get_loss, get_reg, selected_adapters):
    # the composed lora state dict
    adapter = make_linear_combinations_adapter(selected_adapters, weights)

    # reload the model with the new adapter config
    _ = model.load_state_dict(adapter, strict=False)
        
    # minimize the metric
    loss = get_loss(model=model, profile_data=profile_data)

    # L1 regularization term
    metric_val = loss + get_reg(weights)
    
    return metric_val

def get_adapter_prediction(opts, original_model, tokenizer, adapter, generation_config, query_data):
    _ = original_model.load_state_dict(adapter, strict=False)
    # get query prediction
    inputs = tokenizer(query_data[0]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
    outputs = original_model.generate(**inputs, num_beams=opts.generation_num_beams, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
    outputs = outputs.to('cpu')
    tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))
    return tokenized_prediction

if __name__ == "__main__":
    opts = parser.parse_args()
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./experiments', opts.task, f'{dataset_name}_lora_hub')
    log_files_pth = os.path.join(output_dir, 'per_user')

    # Log hyperparameters
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
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)

    from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)
    for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
        if Path(os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json')).is_file():
            continue

        # load user profile and query
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size)
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
        query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, user_id=user_id, data=user_data[user_id])
        txt_label = query_data[0]['target']
        query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

        t0 = time.time()
        # sample N adapters for LoraHub
        selected_adapters_ids = np.random.choice(list(range(len(adapters))), opts.num_lorahub_adapters)
        selected_adapters = [adapters[id_] for id_ in selected_adapters_ids]

        # prepare the minimization function for nevergrad
        get_adapter_loss_partial = partial(get_adapter_loss, opts=opts, output_dir=output_dir, collator=collator, tokenizer=tokenizer)
        get_score_partial = partial(get_score, model=original_model, profile_data=profile_data, get_loss = get_adapter_loss_partial, get_reg=default_l1_regularization, selected_adapters=selected_adapters)
        
        # set up the limit of the weights, and minimize with nevergrad
        instrum = ng.p.Array(
            init=[0] * opts.num_lorahub_adapters,
            upper=[1.5] * opts.num_lorahub_adapters,
            lower=[-1.5] * opts.num_lorahub_adapters,
        )
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=opts.max_inference_step)
        recommendation = optimizer.minimize(get_score_partial, verbosity=1)

        # construct final state dict, load to model and make prediction
        weights = recommendation.value
        final_adapter = make_linear_combinations_adapter(selected_adapters, weights)
        tokenized_prediction = get_adapter_prediction(opts, original_model, tokenizer, final_adapter, generation_config, query_data)
        txt_prediction = tokenizer.decode(tokenized_prediction, skip_special_tokens=True)

        t1 = time.time()

        if not os.path.exists(log_files_pth):
            os.makedirs(log_files_pth)
        with open(os.path.join(log_files_pth, f'{opts.exp_name}results_user_{user_id}.json'), 'w') as file:
            json.dump({
                'user_ids': user_id,
                'final_loss': recommendation.loss,
                'label': txt_label,
                'pred': txt_prediction,
                'chosen_adapters': selected_adapters_ids.tolist(),
                'weights': weights.tolist(),
                'adapters_eval_time': t1-t0
            }, file, indent = 4)