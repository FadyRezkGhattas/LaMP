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

from metrics.utils import get_metrics
from load_adapters import load_adapter
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset
from lora_xs.initialization_utils import find_and_initialize
from prompts.singular_prompts import create_prompt_generator


parser = argparse.ArgumentParser()
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--model_zoo_addr', type=str, default='experiments/LaMP-2/sgd_baseline/r_6_alpha_16_lr_0.01_epochs_20_sch_linear/')
parser.add_argument('--profile_training_ratio', type=float, default=None, help='A ratio to split the profile into training and validation sets. The split ratio If None, no split will be performed.')
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')

def get_adapter_metrics(opts, user_model, tokenizer, data, compute_metrics):
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)
    training_args = Seq2SeqTrainingArguments(
            # trainer basics
            output_dir=opts.model_zoo_addr,
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



def get_adapter_predictions(opts, user_model, tokenizer, data):
    tokenized_predictions = []
    txt_labels = []
    
    for i in tqdm(range(len(data)), leave=False):
        # get query predictions
        inputs = tokenizer(data[i]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
        txt_labels.append(data[i]['target'])
        outputs = user_model.generate(**inputs, num_beams=opts.generation_num_beams, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
        outputs = outputs.to('cpu')
        tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))
        tokenized_predictions.append(tokenized_prediction)
    return tokenized_predictions, txt_labels

if __name__ == '__main__':
    opts = parser.parse_args()
    log_files_pth = os.path.join(opts.model_zoo_addr, 'per_user')

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


    users = os.listdir(os.path.join(opts.model_zoo_addr, 'ckpts'))
    users = [int(x.split('_')[-1]) for x in users]
    
    tokenized_predictions = []
    txt_predictions = []
    txt_labels = []
    user_ids = []
    with tqdm(total=len(users), desc='Processing Users') as pbar:
        for user_id in users:
            user_ids.append(user_id)
            user_model = load_adapter(original_model, os.path.join(opts.model_zoo_addr, 'ckpts', f'user_{user_id}')).to('cuda')
            # Get support performance
            data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size, training_ratio=opts.profile_training_ratio)
            data = convert_to_hf_dataset(data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
            support_results = get_adapter_metrics(opts, user_model, tokenizer, data, compute_metrics)

            # Get query performance
            data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, data=user_data[user_id], truncate_profile_size=opts.truncate_profile_size, training_ratio=opts.profile_training_ratio)
            tokenized_predictions_user, txt_labels_user = get_adapter_predictions(opts, user_model, tokenizer, data)
            txt_predictions_user = tokenizer.batch_decode(tokenized_predictions_user, skip_special_tokens=True)
            
            tokenized_predictions += tokenized_predictions_user
            txt_labels += txt_labels_user
            txt_predictions += txt_predictions_user
            
            if not os.path.exists(log_files_pth):
                os.makedirs(log_files_pth)
            with open(os.path.join(log_files_pth, f'user_{user_id}.json'), 'w') as file:
                json.dump({
                    'user_ids': user_id,
                    'query_label': txt_labels_user,
                    'query_pred': txt_predictions_user,
                    'best_support_metric': support_results[f'support_{best_metric}'],
                }, file, indent = 4)

            pbar.update(1)
    
    tokenized_labels = tokenizer(txt_labels)['input_ids']
    tokenized_labels = np.array([np.pad(torch.tensor(x), (tokenizer.pad_token_id, opts.max_generation_length - len(x))) for x in tokenized_labels])
    results = compute_metrics((tokenized_predictions, tokenized_labels))
    print(results)
    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    with open(os.path.join(opts.model_zoo_addr, f'{dataset_name}.json'), 'w') as file:
        json.dump({
            **results,
            'user_ids':user_ids,
            'labels':txt_labels,
            'preds': txt_predictions
        }, file, indent = 4)