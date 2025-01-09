import os
import json
import yaml
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from torch.utils.data import Subset

from metrics.utils import get_metrics
from prompts.prompts import create_prompt_generator as create_prompt_generator_query
from prompts.singular_prompts import create_prompt_generator as create_prompt_generator_profile
from data.datasets import GeneralSeq2SeqProfilesDataset, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="dev_query")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--subset", choices=['profile', 'query'], default='query', help="Whether to evaluate on user profiles or queries")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--output_dir", default='./experiments')
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")
# If we need to compute performance on samples that only has adapters, then pass a model zoo directory
parser.add_argument('--model_zoo_addr', type=str, default='experiments/LaMP-2/model_zoo/r_6_alpha_16_lr_0.01_epochs_20_sch_linear/ckpts', help='If model zoo directory is provided (a folder with per user adapter), then performance only on users that have an adapter/folder in the model zoo are computed')

if __name__ == "__main__":
    opts = parser.parse_args()
    
    # helper objects
    opts.output_dir = os.path.join(opts.output_dir, opts.task, "evals", opts.exp_name)
    os.makedirs(opts.output_dir, exist_ok=True)

    # Load model, tokenizer, collator, and raw dict data
    model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)

    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

    # Create dataset
    task = opts.task
    greater_is_better = True
    if opts.subset == 'query':
        prompt_generator, _ = create_prompt_generator_query(0, tokenizer=tokenizer)
        dataset = GeneralSeq2SeqDataset(opts.data_addr, True, opts.task, prompt_generator)
    elif opts.subset == 'profile':
        prompt_generator = create_prompt_generator_profile(tokenizer)
        dataset = GeneralSeq2SeqProfilesDataset(opts.task, prompt_generator, data_addr=opts.data_addr, truncate_profile_size=opts.truncate_profile_size)
    
    if opts.model_zoo_addr is not None:
        users = os.listdir(opts.model_zoo_addr)
        users = [int(x.split('_')[-1]) for x in users]
        dataset = Subset(dataset, users)
    print('Dataset Size is:', len(dataset))
    
    # Create metrics
    compute_metrics, best_metric, labels, greater_is_better = get_metrics(task, tokenizer)
    dataset = convert_to_hf_dataset(dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir = opts.output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        eval_accumulation_steps = 1,
        generation_max_length = opts.generation_max_length
    )
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        eval_dataset = dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    results = trainer.evaluate(dataset)
    print(results)

    with open(os.path.join(opts.output_dir,f'{opts.exp_name}.json'), 'w') as file:
        json.dump(results, file, indent = 4)