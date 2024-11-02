import os
import json
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from data.lmdb import LMDBDataset
from metrics.utils import get_metrics
from load_adapters import tensorize_loraxs_adapter
from lora_xs.make_peft_model import make_peft_model
from prompts.singular_prompts import create_prompt_generator
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset

# python diff_adapter_to_zoo_adapter.py --evaluation_zoo 'lmdb_data/LaMP-2-final-pca-512-4x5l-diff-samples' --users_evaluation_logs 'experiments/LaMP-2/dev_questions_merged_stage_4_diff_512_pca_5l_150clusters/'
# python diff_adapter_to_zoo_adapter.py --evaluation_zoo 'lmdb_data/LaMP-2-final-pca-1568-4x3l-diff-samples' --users_evaluation_logs 'experiments/LaMP-2/dev_questions_merged_stage_4_diff_1568_pca_3l_150clusters/'

parser = ArgumentParser()
# Model arguments
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--svd_pth", default='./experiments/fixed_adapter')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
# Dataset Arguments
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
# Zoo arguments
parser.add_argument('--evaluation_zoo', type=str, default='lmdb_data/LaMP-2-final-pca-512-4x3l-diff-samples')
parser.add_argument('--selection_zoo', type=str, default='lmdb_data/LaMP-2-final')
parser.add_argument('--users_evaluation_logs', type=str, default='experiments/LaMP-2/dev_questions_merged_stage_4_diff_512_pca_3l_150clusters/')
opts = parser.parse_args()

def get_adapter_prediction(opts, original_model, tokenizer, adapters, adapter_id, query_data):
    _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
    # get query prediction
    inputs = tokenizer(query_data[0]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
    outputs = original_model.generate(**inputs, num_beams=opts.generation_num_beams, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
    outputs = outputs.to('cpu')
    tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))
    return tokenized_prediction

def get_closest_adapter(adapter, selection_zoo):
    dist = (adapter.unsqueeze(0) - selection_zoo).pow(2).sum(1).sqrt()
    return torch.argmin(dist)

if __name__ == '__main__':
    print("Loading Model, Tokenizer and Preparing PEFT Setup")
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
    original_model = original_model.bfloat16()

    print("Loading metrics")
    compute_metrics, best_metric, txt_labels, greater_is_better = get_metrics(opts.task, tokenizer)

    # Preparing user's data
    print("Loading users data")
    with open(opts.data_addr) as f:
        user_data = json.load(f)
    prompt_generator = create_prompt_generator(tokenizer)
    users_eval_logs_path = os.path.join(opts.users_evaluation_logs, 'per_user')
    users_logs_paths = os.listdir(users_eval_logs_path)
    log_file_prefix = users_logs_paths[0].split('_user')[0]

    print("Loading evaluation and selection zoos")
    evaluation_zoo = LMDBDataset(opts.evaluation_zoo)
    evaluation_zoo = [evaluation_zoo[i] for i in range(len(evaluation_zoo))]
    selection_zoo = LMDBDataset(opts.selection_zoo)
    selection_zoo = [selection_zoo[i] for i in range(len(selection_zoo))]
    selection_zoo = torch.vstack(selection_zoo)

    # Tensorizing selection zoo
    print("Tensorizing selection zoo")
    tensorized_selection_zoo = []
    for i in tqdm(range(len(selection_zoo))):
        tensorized_selection_zoo.append(tensorize_loraxs_adapter(selection_zoo[i], use_bf16=True))

    # Find best adapter in selection zoo and use it for inference
    get_best_adapter_prediction_ = partial(get_adapter_prediction, opts=opts, original_model=original_model, tokenizer=tokenizer, adapters=tensorized_selection_zoo)

    tokenized_predictions = []
    txt_labels = []
    selection_zoo_adapters = []
    for user_id in tqdm(range(len(user_data)), desc='Users', position=1):
        # Prepare user's query data
        query_data = GeneralSeq2SeqProfileDataset(opts.task, prompt_generator, val=True, user_id=user_id, data=user_data[user_id])
        txt_labels.append(query_data[0]['target'])
        query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

        # Load user's selected adapter from evaluation zoo
        with open(os.path.join(users_eval_logs_path, log_file_prefix+f'_user_{user_id}.json')) as f:
            user_logs = json.load(f)
        selected_adapter_id = user_logs['best_adapter_ids']

        # Find closest adapter in selection zoo
        best_adapter_id = get_closest_adapter(evaluation_zoo[selected_adapter_id], selection_zoo)
        selection_zoo_adapters.append(best_adapter_id)
        
        # Evaluate closest adapter retrieved from selection zoo
        tokenized_prediction = get_best_adapter_prediction_(adapter_id=best_adapter_id, query_data=query_data)
        tokenized_predictions.append(tokenized_prediction)

    txt_predictions = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)
    tokenized_labels = tokenizer(txt_labels)['input_ids']
    tokenized_labels = np.array([np.pad(torch.tensor(x), (tokenizer.pad_token_id, opts.max_generation_length - len(x))) for x in tokenized_labels])
    results = compute_metrics((tokenized_predictions, tokenized_labels))
    print(results)
    with open(os.path.join(opts.users_evaluation_logs, f'select_from_zoo_{os.path.basename(opts.selection_zoo)}.json'), 'w') as file:
            json.dump({
                'labels': txt_labels,
                'preds': txt_predictions,
                'selection_zoo_adapters': selection_zoo_adapters,
            }, file, indent = 4)