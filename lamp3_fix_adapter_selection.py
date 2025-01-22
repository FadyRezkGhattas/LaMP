# LaMP-3 had a bug in the adapter selection stage. After shortlisting best-15 adapters using min loss, we compute the dataset ``best metric`` on these 15 adapters using generation.
# On the ``best metric``, we choose the best adapter. Best metric was ROUGE and Accuracy for LaMP-2 and 5. For LaMP-3, it was an error. Therefore, best adapter from 15 is the one achieving lowest metric.
# Nevertheless, the evaluation script was always choosing max.

# To fix, we can load every user's json log and make a one to one correspondence between adapter loss and adapter id. 
# We have the losses over a chosen cluster and the chosen adapter id which tells us which cluster was chosen.
# By ranking losses, we can know the IDs of top-15. This produces a correspondence between adapter id and ``best metric``
# Then we can update the best adapter chosen and re-evaluate

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
parser.add_argument("--results_addr", default="./experiments/LaMP-3/dev_questions_merged_stage_4_model_zoo_150clusters_trnct_256")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_3/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-3/finetune_all_train_user_profiles/checkpoint-117500')
parser.add_argument("--svd_pth", default='./experiments/fixed_adapter')
parser.add_argument("--use_bf16", default=True)
parser.add_argument("--task", default='LaMP-3')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")

# diffusion model and model zoo
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-3-final')
parser.add_argument('--lmdb_clusters', type=str, default='lmdb_data/LaMP-3-final/150_clusters.json', help='If provided, the medoids are used to chose a cluster whose adapters are evaluated for a user.')

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
    return user_log, user_file_path

def get_adapter_prediction(opts, original_model, tokenizer, query_data, generation_config):
    tokenized_predictions = []
    for i in range(len(query_data)):
        # get query predictions
        inputs = tokenizer(query_data[i]["source"], truncation=True, max_length=opts.max_length, return_tensors="pt").to('cuda')
        outputs = original_model.generate(**inputs, num_beams=opts.generation_num_beams, generation_config=generation_config, max_new_tokens=opts.max_generation_length)
        outputs = outputs.to('cpu')
        tokenized_prediction = F.pad(outputs[0], (tokenizer.pad_token_id, opts.max_generation_length - len(outputs[0])))
        tokenized_predictions.append(tokenized_prediction)
    return tokenized_predictions

if __name__ == '__main__':
    opts = parser.parse_args()
    
    with open(os.path.join(opts.results_addr, "hyperparameters_fix_lamp_3_adapter_selection.json"), 'w') as f:
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

    # load clusters
    with open(opts.lmdb_clusters) as f:
        lmdb_clusters = json.load(f)
    medoids_ids = lmdb_clusters['medoids']
    clusters_ids = lmdb_clusters['clusters']

    # text data processors
    prompt_generator = create_prompt_generator(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)

    print("searching for best cluster")
    for user_id in tqdm(range(len(user_data)), leave=True, desc='Users', position=0):
        # load user log
        user_log, user_log_path = get_user_json_results(opts, user_id)

        # load user query data
        query_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=True, user_id=user_id, data=user_data[user_id])
        assert len(query_data) == 1
        txt_labels_user = query_data[0]['target']
        query_data = convert_to_hf_dataset(query_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
        assert txt_labels_user == user_log["label"]

        # find the chosen medoid index for the user
        bugged_best_adapter_id = user_log['best_adapter_ids']
        best_medoid_id = [i for i, cluster_ids in enumerate(clusters_ids) if bugged_best_adapter_id in cluster_ids]
        assert len(best_medoid_id) == 1 # sanity check: should find one cluster only
        best_medoid_id = best_medoid_id[0]

        # retrieve the adapter IDs belonging to the cluster of the found medoid
        zoo_adapter_ids_in_chosen_cluster = clusters_ids[best_medoid_id]
        assert len(zoo_adapter_ids_in_chosen_cluster) == len(user_log['user_train_perfs']) # sanity check: assert that size of found cluster and size of support losses are equal

        # rank the user support losses to find top 15 adapters in the cluster
        position_of_best_15_adapters_in_cluster = np.argsort(user_log['user_train_perfs'])[:15]
        # get the adapter indices in the model zoo
        best_15_adapters_idx = [zoo_adapter_ids_in_chosen_cluster[i] for i in position_of_best_15_adapters_in_cluster]

        # now we have a 15 best adapter IDs and thei ``best metric`` list
        # if we find adapter with highest loss, we would reproduce the bug successfully
        best_15_adapters_idx_perf = user_log['best_15_adapters_accuracies']
        assert bugged_best_adapter_id == best_15_adapters_idx[np.argmax(best_15_adapters_idx_perf)]

        # find true best adapter
        fixed_best_adapter_id = best_15_adapters_idx[np.argmin(best_15_adapters_idx_perf)]

        # load best adapter
        _ = original_model.load_state_dict(adapters[fixed_best_adapter_id], strict=False)

        # get adapter prediction
        tokenized_predictions = get_adapter_prediction(opts, original_model, tokenizer, query_data, generation_config)
        txt_prediction = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)
        assert len(txt_prediction) == 1
        txt_prediction = txt_prediction[0]

        # update adapter id, and preds in user log and overwrite
        print(f'old pred={user_log["pred"]} and new pred={txt_prediction}')
        print(f'old adapter={bugged_best_adapter_id} and new adapter={fixed_best_adapter_id}')
        user_log['best_adapter_ids'] = fixed_best_adapter_id
        user_log['pred'] = txt_prediction

        with open(user_log_path, 'w') as f:
            json.dump(user_log, f, indent = 4)