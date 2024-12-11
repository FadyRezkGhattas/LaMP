import os
import json
import yaml
import copy
import argparse
from rich import print

from accelerate import Accelerator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments
from trainers.mezo import MezoTrainer, MezoArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq

from metrics.utils import get_metrics
from lora_xs.make_peft_model import make_peft_model
from utils.utils import CSVLogger, opts_to_exp_name
from prompts.singular_prompts import create_prompt_generator

from data.datasets import GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_prefix', type=str, default="mezo/")
parser.add_argument("--num_tasks", type=int, default=-1, help="Train for fixed number of tasks. If -1, then train for all available tasks.")
parser.add_argument("--from_user_id", type=int, default=0, help="Train model starting from this user index.")
parser.add_argument("--to_user_id", type=int, default=-1, help="Terminate training at this user index. If -1, train until end of available users.")
parser.add_argument('--truncate_profile_size', type=int, default=-1, help='if > 0, then the profile size is truncated to max of given value.')
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/dev_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--svd_pth", default='./experiments/fixed_adapter')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--output_dir", default='./experiments')
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 32)
parser.add_argument("--learning_rate", type = float, default = 0.01)
parser.add_argument("--num_train_epochs", type = int, default = 120)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--do_eval", type = bool, default = False)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":
    opts = parser.parse_args()
    accelerator = Accelerator()
    
    # helper objects
    exp_name = opts.exp_prefix + opts_to_exp_name(opts)
    opts.output_dir = os.path.join(opts.output_dir, opts.task, exp_name)
    logger = CSVLogger(opts.output_dir, exp_name)

    # Log hyperparameters
    os.makedirs(opts.output_dir, exist_ok=True)
    with open(os.path.join(opts.output_dir, "hyperparameters.json"), 'w') as f:
        json.dump(vars(opts), f)

    # Load all users data
    with open(opts.data_addr) as f:
        data = json.load(f)

    # Load model, tokenizer, collator, and raw dict data
    original_model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)

    # prepare model for PEFTing
    original_model = make_peft_model(opts, original_model)

    # Create metrics
    task = opts.task
    greater_is_better = True
    compute_metrics, best_metric, labels, greater_is_better = get_metrics(task, tokenizer)

    task_counter = 0
    from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(data)
    for user_id in range(from_, to_):
        # If adapter exists for user, skip
        ckpt_path = os.path.join(opts.output_dir, 'ckpts', "user_" + str(user_id))
        if os.path.isfile(os.path.join(ckpt_path, 'adapter_model.safetensors')):
            continue
        # Copy the model for the user and create an appropropriate data collator
        model = copy.deepcopy(original_model)
        collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

        # Create datasets
        prompt_generator = create_prompt_generator(tokenizer)

        # Profile data for training
        train_dataset = GeneralSeq2SeqProfileDataset(task, prompt_generator, data=data[user_id], truncate_profile_size=opts.truncate_profile_size)
        # Query sample to eval on
        test_dataset = GeneralSeq2SeqProfileDataset(task, prompt_generator, data=data[user_id], val=True)
        
        train_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
        eval_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
        test_dataset = convert_to_hf_dataset(test_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)

        training_args = MezoArguments(
            # trainer basics
            output_dir=opts.output_dir,
            do_eval = opts.do_eval,
            evaluation_strategy = "steps" if opts.do_eval else "no",
            eval_steps=5,
            # parallelization args
            per_device_train_batch_size = opts.per_device_batch_size,
            per_device_eval_batch_size = opts.per_device_batch_size,
            gradient_accumulation_steps = 1, # mezo does not support grad accumulation
            # optimizer args
            learning_rate = opts.learning_rate,
            num_train_epochs = opts.num_train_epochs,
            lr_scheduler_type = opts.lr_scheduler_type,
            warmup_ratio = opts.warmup_ratio,
            # generation args
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate = True,
            generation_max_length = opts.generation_max_length,
            # logging strategy
            logging_steps = 1,
            report_to="tensorboard",
            # mezo args
            trainer='zo',
        )

        trainer = MezoTrainer(
            model = model,
            args = training_args,
            data_collator = collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            compute_metrics = compute_metrics
        )

        # Get performance pre-training
        pretrain_train_metrics = trainer.evaluate(train_dataset)
        pretrain_train_metrics = {k.replace("eval", "pre_finetuning"): v for k, v in pretrain_train_metrics.items()}

        pretrain_eval_metrics = trainer.evaluate(test_dataset)
        pretrain_eval_metrics = {k.replace("eval", "pretrain_eval"): v for k, v in pretrain_eval_metrics.items()}

        # Train model
        trainer.train()

        # get performance post-training
        posttrain_train_metrics = trainer.evaluate(train_dataset)
        posttrain_train_metrics = {k.replace("eval", "post_finetuning"): v for k, v in posttrain_train_metrics.items()}

        posttrain_test_metrics = trainer.evaluate(test_dataset)
        posttrain_test_metrics = {k.replace("eval", "posttrain_eval"): v for k, v in posttrain_test_metrics.items()}

        if accelerator.is_main_process:
            # Log results
            logging_data = {'user_id': user_id, 'profile_size':len(train_dataset),  **pretrain_train_metrics, **pretrain_eval_metrics, **posttrain_train_metrics, **posttrain_test_metrics}
            logger.log(trainer=None, extra_data=logging_data)
            task_counter += 1

            # Save Adapter
            for param in model.parameters(): param.data = param.data.contiguous()
            model.save_pretrained(ckpt_path)

        if task_counter == opts.num_tasks:
            break