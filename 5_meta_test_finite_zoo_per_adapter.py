import os
import yaml
import json
import argparse
from tqdm import tqdm

import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers.trainer_callback import PrinterCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.lmdb import LMDBDataset
from metrics.utils import get_metrics
from load_adapters import tensorize_loraxs_adapter
from lora_xs.initialization_utils import find_and_initialize
from prompts.singular_prompts import create_prompt_generator
from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_prefix", default="")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--per_device_batch_size", type = int, default = 32)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--max_generation_length", type = int, default = 128)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-2-v1')
parser.add_argument('--num_tasks', type=int, default=-1, help='total number of tasks to evaluate model zoo on. If -1, all users are evaluated.')
parser.add_argument('--early_stop', type=int, default=1e10, help='how many steps to wait for performance to not improve before skipping the rest of the model zoo')

if __name__ == '__main__':
    opts = parser.parse_args()
    dataset_name = opts.data_addr.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./experiments', opts.task, f'{dataset_name}_model_zoo_as_finite_hypothesis')

    print("Loading Model")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="./cache")
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
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
        writer=None, reconstruct_config=reconstr_config
        )
    original_model.print_trainable_parameters()
    original_model = original_model.to('cuda')
    original_model.eval()
    for name, param in original_model.named_parameters():
        param.contiguous()

    print("loading Dataset and Metrics")
    # Load all users data    
    task = opts.task
    compute_metrics, best_metric, txt_labels, greater_is_better = get_metrics(task, tokenizer)
    with open(opts.data_addr) as f:
        user_data = json.load(f)

    print("Prepare Users Datasets")
    prompt_generator = create_prompt_generator(tokenizer)
    num_tasks = opts.num_tasks if opts.num_tasks != -1 else len(user_data)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = original_model)
    profiles_data = []
    queries_data = []
    for user_id in tqdm(range(num_tasks), desc='User', position=0):
        profile_data = GeneralSeq2SeqProfileDataset(task, prompt_generator, val=False, user_id=user_id, data=user_data[user_id])
        profile_data = convert_to_hf_dataset(profile_data, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
        profiles_data.append(profile_data)

    print("Loading and Tensorizing Model Zoo")
    model_zoo = LMDBDataset(opts.lmdb_addr)

    print("Tensorizing model zoo")
    adapters = []
    for i in range(len(model_zoo)):
        adapters.append(tensorize_loraxs_adapter(model_zoo[i]))

    print("Begin Evaluation")
    for adapter_id in tqdm(range(len(adapters)), leave=False, desc='Adapters', position=1):
        results = {}
        _ = original_model.load_state_dict(adapters[adapter_id], strict=False)
        training_args = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            do_eval = True,
            per_device_eval_batch_size = opts.per_device_batch_size,
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate = True,
            eval_accumulation_steps = 1,
            generation_max_length = opts.max_generation_length,
            disable_tqdm=True
        )
        trainer = Seq2SeqTrainer(
            model = original_model,
            args = training_args,
            data_collator = collator,
            eval_dataset = None,
            tokenizer = tokenizer,
            compute_metrics = compute_metrics
        )
        trainer.remove_callback(PrinterCallback)
        adapter_results = []
        for user_id in tqdm(range(num_tasks), desc='User', position=0):
            profile_data = profiles_data[user_id]              
            metrics = trainer.evaluate(profile_data)
            results[user_id] = metrics

        with open(os.path.join(output_dir, 'per_adapter', f'{opts.exp_prefix}results_adapter_{adapter_id}.json'), 'w') as file:
            json.dump(results, file, indent = 4)