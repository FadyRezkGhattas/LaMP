import os
import yaml
import json
import copy
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

from utils.utils import CSVLogger
from data.lmdb import LMDBDataset
from metrics.utils import get_metrics
from load_adapters import tensorize_loraxs_adapter
from lora_xs.initialization_utils import find_and_initialize
from prompts.singular_prompts import create_prompt_generator
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--lmdb_addr", default="lmdb_data/LaMP-2-v1")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--per_device_batch_size", default=64)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--model_zoo_addr', type=str, default='experiments/LaMP-2/sgd_baseline/r_6_alpha_16_lr_0.01_epochs_20_sch_linear/')
parser.add_argument("--from_user_id", type=int, default=383, help="Train model starting from this user index.")
parser.add_argument("--to_user_id", type=int, default=384, help="Terminate training at this user index. If -1, train until end of available users.")
parser.add_argument("--experiment_type", default='eval_adapters', choices=['eval_adapters', 'gd', 'zs'])

def split_data(user_data):
    """
    Splits the data into a training set and a test set based on class counts.

    Args:
        user_data (dict): The input data containing a user profile.

    Returns:
        tuple: A tuple containing the train_data and test_data.
    """

    user_data_profile = user_data['profile']
    data_by_tag = {}
    for item in user_data_profile:
        output_tag = item['tag']
        if output_tag not in data_by_tag:
            data_by_tag[output_tag] = []
        data_by_tag[output_tag].append(item)

    # Convert the dictionary values to a list of lists
    data_by_tag_list = list(data_by_tag.values())

    # Split the data into training and test sets based on the class counts
    train_data = []
    test_data = []
    for cls in data_by_tag_list:
        # If the class has only one sample, add it to the training set
        if len(cls) == 1:
            train_data.append(cls[0])
        else:
            split_size = len(cls) // 2
            train_data += cls[:split_size]
            test_data += cls[split_size:]
    
    user_data['profile'] = train_data
    train_data = copy.deepcopy(user_data)

    user_data['profile'] = test_data
    test_data = copy.deepcopy(user_data)
    return train_data, test_data

if __name__ == '__main__':
    opts = parser.parse_args()

    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./experiments', opts.task, f'{dataset_name}_adapter_selection_analysis')

    print("Loading Model")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="./cache")
    generation_config = GenerationConfig.from_pretrained(opts.model_name)

    # prepare model for PEFTing
    if opts.experiment_type != 'zs':
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

    print("loading Dataset and Metrics")
    # Load all users data    
    task = opts.task
    compute_metrics, best_metric, txt_labels, greater_is_better = get_metrics(task, tokenizer)
    prompt_generator = create_prompt_generator(tokenizer)
    with open(opts.data_addr) as f:
        user_data = json.load(f)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)

    def eval_adapter():  
        print("Loading and Tensorizing Model Zoo")
        model_zoo = LMDBDataset(opts.lmdb_addr)

        print("Tensorizing model zoo")
        adapters = []
        for i in range(len(model_zoo)):
            adapters.append(tensorize_loraxs_adapter(model_zoo[i]))

        from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)
        for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
            logger = CSVLogger(output_dir, f'adapters_zoo_user_{user_id}_support_eval_with_greedy')

            train_data, test_data = split_data(user_data[user_id])
            
            support_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=train_data)
            support_data = convert_to_hf_dataset(support_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
            query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=test_data)
            query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

            training_args = Seq2SeqTrainingArguments(
                output_dir = output_dir,
                do_eval = True,
                per_device_eval_batch_size = opts.per_device_batch_size,
                generation_num_beams=1,
                predict_with_generate = True,
                eval_accumulation_steps = 1,
                generation_max_length = opts.max_generation_length,
                disable_tqdm=True,
                bf16=True
            )
            support_evaluator = Seq2SeqTrainer(
                model = original_model,
                args = training_args,
                data_collator = collator,
                eval_dataset = support_data,
                tokenizer = tokenizer,
                compute_metrics = compute_metrics,
                # preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
            support_evaluator.remove_callback(PrinterCallback)
            
            eval_args = Seq2SeqTrainingArguments(
                output_dir = output_dir,
                do_eval = True,
                per_device_eval_batch_size = opts.per_device_batch_size,
                generation_num_beams = opts.generation_num_beams,
                predict_with_generate = True,
                eval_accumulation_steps = 1,
                generation_max_length = opts.max_generation_length,
                disable_tqdm=True,
                bf16=True
            )
            query_evaluator = Seq2SeqTrainer(
                model = original_model,
                args = eval_args,
                data_collator = collator,
                eval_dataset = query_data,
                tokenizer = tokenizer,
                compute_metrics = compute_metrics
            )
            query_evaluator.remove_callback(PrinterCallback)

            for adapter_id in tqdm(range(len(adapters)), leave=False, desc='Adapters', position=1):
                _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
                    
                support_results = support_evaluator.evaluate(support_data, metric_key_prefix="support")
                query_results = query_evaluator.evaluate(query_data, metric_key_prefix="query")

                logging_data = {'user_id': user_id, 'support_size': len(support_data), 'query_size': len(query_data),  **support_results, **query_results}
                logger.log(trainer=None, extra_data=logging_data)
    
    def sgd_adapt():
        from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)
        for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
            logger = CSVLogger(output_dir, f'sgd_user_{user_id}')

            train_data, test_data = split_data(user_data[user_id])
            
            support_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=train_data)
            support_data = convert_to_hf_dataset(support_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
            query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=test_data)
            query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

            training_args = Seq2SeqTrainingArguments(
                output_dir = output_dir,
                do_train=True,
                do_eval = True,
                bf16=True,
                # parallelization args
                per_device_train_batch_size = 32,
                per_device_eval_batch_size = opts.per_device_batch_size,
                gradient_accumulation_steps = 1,
                # optimizer args
                learning_rate = 0.01,
                num_train_epochs = 20,
                lr_scheduler_type = 'linear',
                warmup_ratio = 0.05,
                # generation args
                generation_num_beams = opts.generation_num_beams,
                predict_with_generate = True,
                eval_accumulation_steps = 1,
                generation_max_length = opts.max_generation_length,
            )

            trainer = Seq2SeqTrainer(
                model = original_model,
                args = training_args,
                data_collator = collator,
                train_dataset=support_data,
                eval_dataset = query_data,
                tokenizer = tokenizer,
                compute_metrics = compute_metrics
            )
            
            trainer.train()
            support_results = trainer.evaluate(support_data, metric_key_prefix="support")
            query_results = trainer.evaluate(query_data, metric_key_prefix="query")

            logging_data = {'user_id': user_id, 'support_size': len(support_data), 'query_size': len(query_data),  **support_results, **query_results}
            logger.log(trainer=None, extra_data=logging_data)

    def eval_zs():
        from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(user_data)
        for user_id in tqdm(range(from_, to_), leave=True, desc='Users', position=0):
            logger = CSVLogger(output_dir, f'zs_user_{user_id}')

            train_data, test_data = split_data(user_data[user_id])
            
            support_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=train_data)
            support_data = convert_to_hf_dataset(support_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
            query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=test_data)
            query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

            training_args = Seq2SeqTrainingArguments(
                output_dir = output_dir,
                do_eval = True,
                per_device_eval_batch_size = opts.per_device_batch_size,
                generation_num_beams = opts.generation_num_beams,
                predict_with_generate = True,
                eval_accumulation_steps = 1,
                generation_max_length = opts.max_generation_length,
                disable_tqdm=True,
                bf16=True
            )
            trainer = Seq2SeqTrainer(
                model = original_model,
                args = training_args,
                data_collator = collator,
                train_dataset=support_data,
                eval_dataset = query_data,
                tokenizer = tokenizer,
                compute_metrics = compute_metrics
            )
            trainer.remove_callback(PrinterCallback)
                                
            support_results = trainer.evaluate(support_data, metric_key_prefix="support")
            query_results = trainer.evaluate(query_data, metric_key_prefix="query")

            logging_data = {'user_id': user_id, 'support_size': len(support_data), 'query_size': len(query_data),  **support_results, **query_results}
            logger.log(trainer=None, extra_data=logging_data)
        
    if opts.experiment_type == 'eval_adapters':
        eval_adapter()
    elif opts.experiment_type == 'gd':
        sgd_adapt()
    elif opts.experiment_type == 'zs':
        eval_zs()