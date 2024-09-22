import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from metrics.classification_metrics import create_metric_f1_accuracy
from data.datasets import get_all_labels, GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset, train_val_split
from prompts.singular_prompts import create_prompt_generator
from peft import get_peft_model, PromptTuningConfig
import json
from utils import CSVLogger, opts_to_exp_name

parser = argparse.ArgumentParser()
parser.add_argument('--exp_prefix', type=str, default="")
parser.add_argument("--num_tasks", type=int, default=20)
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/train_questions_merged.json")
parser.add_argument("--model_name", default='google/flan-t5-base')
parser.add_argument("--num_virtual_tokens", type=int, default=2)
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--output_dir", default='./experiments')
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--learning_rate", type = float, default = 5e-5)
parser.add_argument("--weight_decay", type = float, default = 0.0001)
parser.add_argument("--num_train_epochs", type = int, default = 10)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":

    opts = parser.parse_args()
    
    with open(opts.data_addr) as f:
        data = json.load(f)

    # helper objects
    exp_name = opts.exp_prefix + opts_to_exp_name(opts)
    opts.output_dir = os.path.join(opts.output_dir, opts.task, exp_name)
    logger = CSVLogger(opts.output_dir, exp_name)

    task_counter = 0
    for user_id in range(len(data)):
        # Load model, tokenizer, collator, and raw dict data
        model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
        collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

        # Create datasets and metrics
        task = opts.task
        prompt_generator = create_prompt_generator(tokenizer)
        greater_is_better = True
        if task == "LaMP-2":
            user_dataset, labels = GeneralSeq2SeqProfileDataset(task, prompt_generator, data=data[user_id]), get_all_labels(task)
            if len(user_dataset) < 60:
                continue
            train_dataset, eval_dataset = train_val_split(user_dataset, val_size=0.2)
            compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
            best_metric = "accuracy"
        
        train_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
        eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)

        # prepare model for PEFTing
        peft_config = PromptTuningConfig(
            task_type="SEQ_2_SEQ_LM",
            num_virtual_tokens=opts.num_virtual_tokens,
            prompt_tuning_init="RANDOM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


        training_args = Seq2SeqTrainingArguments(
            output_dir = opts.output_dir,
            do_train = True,
            do_eval = True,
            evaluation_strategy = "epoch",
            per_device_train_batch_size = opts.per_device_batch_size,
            per_device_eval_batch_size = opts.per_device_batch_size,
            gradient_accumulation_steps = opts.gradient_accumulation_steps,
            learning_rate = opts.learning_rate,
            weight_decay = opts.weight_decay,
            num_train_epochs = opts.num_train_epochs,
            lr_scheduler_type = opts.lr_scheduler_type,
            warmup_ratio = opts.warmup_ratio,
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate = True,
            save_strategy = "epoch",
            logging_steps = 50,
            eval_accumulation_steps = 1,
            generation_max_length = opts.generation_max_length,
            load_best_model_at_end = True,
            metric_for_best_model = best_metric,
            greater_is_better = greater_is_better,
            save_total_limit=1,
            save_steps=50,
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
        logger.log(trainer)
        task_counter += 1
        if task_counter == opts.num_tasks:
            break