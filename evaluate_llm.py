import os
import json
import yaml
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq

from prompts.singular_prompts import create_prompt_generator as create_prompt_generator_profile
from prompts.prompts import create_prompt_generator as create_prompt_generator_query
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from data.datasets import get_all_labels, GeneralSeq2SeqProfilesDataset, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="dev_query")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--subset", choices=['profile', 'query'], default='query', help="Whether to evaluate on user profiles or queries")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--output_dir", default='./experiments')
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":
    opts = parser.parse_args()
    
    # helper objects
    opts.exp_name = "evals/" + opts.exp_name
    opts.output_dir = os.path.join(opts.output_dir, opts.task, opts.exp_name)
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
        dataset = GeneralSeq2SeqProfilesDataset(opts.task, prompt_generator, data_addr=opts.data_addr)
    
    # Create metrics
    labels = get_all_labels(task)
    if task == "LaMP-2":
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
        best_metric = "accuracy"
    elif task == "LaMP-3":
        compute_metrics = create_metric_mae_rmse(tokenizer = tokenizer, all_labels = labels)
        best_metric = "mae"
        greater_is_better = False
    elif task == "LaMP-4":
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP-5":
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    else:
        raise ValueError(f"Task {task} not supported")

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