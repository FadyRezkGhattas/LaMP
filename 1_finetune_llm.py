import os
import json
import argparse
from rich import print

from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

from metrics.utils import get_metrics

from peft import LoraConfig, get_peft_model
from prompts.prompts import create_prompt_generator as create_prompt_generator_val
from prompts.singular_prompts import create_prompt_generator as create_prompt_generator_train
from data.datasets import GeneralSeq2SeqDataset, GeneralSeq2SeqProfilesDataset, get_all_labels, create_preprocessor, convert_to_hf_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--train_data", default="./data_raw/user/LaMP_2/train_questions_merged.json")
parser.add_argument("--validation_data", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--test_data", default="")
parser.add_argument("--model_name", default='google/flan-t5-base')
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--output_dir", default='./experiments/LaMP-2/finetune_all_train_user_profiles')
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 64)
parser.add_argument("--learning_rate", type = float, default = 5e-5)
parser.add_argument("--weight_decay", type = float, default = 0.0001)
parser.add_argument("--num_train_epochs", type = int, default = 50)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--num_retrieved", type = int, default=4)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument("--use_lora", default=False, type=bool)
parser.add_argument("--lora_alpha", default=16, type=int)
parser.add_argument("--rank", default=6, type=int)

if __name__ == "__main__":
    opts = parser.parse_args()
    os.makedirs(opts.output_dir, exist_ok=True)
    with open(os.path.join(opts.output_dir, "hyperparameters.json"), 'w') as f:
        json.dump(vars(opts), f)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

    # prepare model for PEFTing
    if opts.use_lora:
        config = LoraConfig(
            r=opts.rank,
            target_modules=["q", "v"],
            task_type="SEQ_2_SEQ_LM", # assuming a decoder-only model in this example
            lora_alpha=opts.lora_alpha,
            use_rslora=True
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()


    print("[bold magenta]Step 1: Loading data and metrics...[/bold magenta]")
    task = opts.task
    prompt_generator_train = create_prompt_generator_train(tokenizer)
    prompt_generator_val, _ = create_prompt_generator_val(0, tokenizer=tokenizer)

    # Load datasets
    print("[bold magenta]Step 2 (a): Loading data and metrics...[/bold magenta]")
    train_dataset, labels = GeneralSeq2SeqProfilesDataset(task, prompt_generator_train, data_addr=opts.train_data), get_all_labels(task)
    eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, True, task, prompt_generator_val)
    if opts.test_data:
        test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator_val)

    compute_metrics, best_metric, labels, greater_is_better = get_metrics(task, tokenizer)
    
    print("[bold magenta]Step 2(b): Preprocessing data...[/bold magenta]")
    print("Processing train data")
    print(f"Train Split Size: {len(train_dataset)}")
    train_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
    print("Processing eval data")
    print(f"Eval Split Size: {len(eval_dataset)}")
    eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
    if opts.test_data:
        test_dataset = convert_to_hf_dataset(test_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
    
    print("[bold magenta]Step 3: Preparing training arguments and launching experiment[/bold magenta]")
    training_args = Seq2SeqTrainingArguments(
        # trainer basics
        output_dir = opts.output_dir,
        do_train = True,
        do_eval = True,
        evaluation_strategy = "steps",
        eval_steps=500,
        # parallelization args
        per_device_train_batch_size = opts.per_device_batch_size,
        per_device_eval_batch_size = opts.per_device_batch_size,
        gradient_accumulation_steps = opts.gradient_accumulation_steps,
        # optimizer args
        learning_rate = opts.learning_rate,
        weight_decay = opts.weight_decay,
        num_train_epochs = opts.num_train_epochs,
        lr_scheduler_type = opts.lr_scheduler_type,
        warmup_ratio = opts.warmup_ratio,
        # generation args
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        generation_max_length = opts.generation_max_length,
        # logging strategy
        logging_steps = 50,
        eval_accumulation_steps = 1,
        load_best_model_at_end = True,
        metric_for_best_model = best_metric,
        greater_is_better = greater_is_better,
        save_total_limit=1,
        save_strategy = "steps",
        save_steps=500,
        report_to="tensorboard"
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    trainer.train()

    if opts.test_data:
        results = trainer.evaluate(test_dataset)
        print(results)

        with open(os.join(opts.output_dir,'results_output.json'), 'w') as file:
            json.dump(results, file, indent = 4)