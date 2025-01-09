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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from metrics.utils import get_metrics
from load_adapters import load_adapter
from data.datasets import GeneralSeq2SeqDataset
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

if __name__ == '__main__':
    opts = parser.parse_args()

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
    dataset = GeneralSeq2SeqDataset(opts.data_addr, use_profile=False, task=task, create_prompt=None)

    users = os.listdir(os.path.join(opts.model_zoo_addr, 'ckpts'))
    users = [int(x.split('_')[-1]) for x in users]
    
    tokenized_predictions = []
    txt_labels = []
    user_ids = []
    with tqdm(total=len(users), desc='Processing Users') as pbar:
        for user_id in users:
            user_model = load_adapter(original_model, os.path.join(opts.model_zoo_addr, 'ckpts', f'user_{user_id}')).to('cuda')
            item = dataset[user_id]
            inputs = tokenizer(item["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
            txt_labels.append(item['target'])
            outputs = user_model.generate(**inputs, num_beams=4, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
            outputs = outputs.to('cpu')
            # tokenized_predictions.append(outputs[0])
            tokenized_predictions.append(F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0]))))
            user_ids.append(item['id'])
            pbar.update(1)

    txt_predictions = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)
    
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