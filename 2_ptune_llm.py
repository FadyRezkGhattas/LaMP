import os
import json
import yaml
import copy
import argparse
from rich import print

from peft import get_peft_model, LoraConfig
from lora_xs.initialization_utils import find_and_initialize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq

from utils import CSVLogger, opts_to_exp_name
from prompts.singular_prompts import create_prompt_generator
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from data.datasets import get_all_labels, GeneralSeq2SeqProfileDataset, create_preprocessor, convert_to_hf_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_prefix', type=str, default="lora_xs/")
parser.add_argument("--num_tasks", type=int, default=-1, help="Train for fixed number of tasks. If -1, then train for all available tasks.")
parser.add_argument("--from_user_id", type=int, default=0, help="Train model starting from this user index.")
parser.add_argument("--to_user_id", type=int, default=-1, help="Terminate training at this user index. If -1, train until end of available users.")
parser.add_argument("--data_addr", default="./data_raw/user/LaMP_2/train_questions_merged.json")
parser.add_argument("--model_name", default='./experiments/LaMP-2/finetune_all_train_user_profiles/checkpoint-32000')
parser.add_argument("--rank", type=int, default=6)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--task", default='LaMP-2')
parser.add_argument("--output_dir", default='./experiments')
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 32)
parser.add_argument("--learning_rate", type = float, default = 0.01)
parser.add_argument("--num_train_epochs", type = int, default = 20)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":
    opts = parser.parse_args()
    
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
    config = LoraConfig(
        r=opts.rank,
        target_modules=["q", "v"],
        task_type="SEQ_2_SEQ_LM", # assuming a decoder-only model in this example
        lora_alpha=opts.lora_alpha,
        use_rslora=True
        )
    original_model = get_peft_model(original_model, config)

    with open("lora_xs/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    adapter_name = "default"  # assuming a single LoRA adapter per module should be transformed to LoRA-XS
    peft_config_dict = {adapter_name: config}
    reconstr_config['svd']['rank'] = opts.rank
    find_and_initialize(
        original_model, peft_config_dict, adapter_name=adapter_name, reconstr_type='svd',
        writer=None, reconstruct_config=reconstr_config
        )
    original_model.print_trainable_parameters()

    task_counter = 0
    from_, to_ = opts.from_user_id, opts.to_user_id if opts.to_user_id != -1 else len(data)
    for user_id in range(from_, to_):
        # Copy the model for the user and create an appropropriate data collator
        model = copy.deepcopy(original_model)
        collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

        # Create datasets and metrics
        task = opts.task
        prompt_generator = create_prompt_generator(tokenizer)
        greater_is_better = True
        train_dataset, labels = GeneralSeq2SeqProfileDataset(task, prompt_generator, data=data[user_id]), get_all_labels(task)
        test_dataset = GeneralSeq2SeqProfileDataset(task, prompt_generator, data=data[user_id], val=True)
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
        
        train_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
        eval_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)
        test_dataset = convert_to_hf_dataset(test_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = tokenizer.model_max_length), batched=True)

        training_args = Seq2SeqTrainingArguments(
            # trainer basics
            output_dir = opts.output_dir,
            do_train = True,
            do_eval = True,
            evaluation_strategy = "steps",
            eval_steps=5,
            # parallelization args
            per_device_train_batch_size = opts.per_device_batch_size,
            per_device_eval_batch_size = opts.per_device_batch_size,
            gradient_accumulation_steps = opts.gradient_accumulation_steps,
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

        # Get performance pre-training
        pre_train_metrics = trainer.evaluate(train_dataset)
        pre_train_metrics = {k.replace("eval", "pre_finetuning"): v for k, v in pre_train_metrics.items()}

        # Train model
        trainer.train()

        # get performance post-training
        post_train_metrics = trainer.evaluate(train_dataset)
        post_train_metrics = {k.replace("eval", "post_finetuning"): v for k, v in post_train_metrics.items()}

        # Test dataset
        test_metrics = trainer.evaluate(test_dataset)

        # Log results
        logger.log(trainer=None, extra_data={'user_id': user_id, **pre_train_metrics, **post_train_metrics, **test_metrics})
        task_counter += 1

        # Save Adapter
        for param in model.parameters(): param.data = param.data.contiguous()
        model.save_pretrained(os.path.join(opts.output_dir, 'ckpts', "user_" + str(user_id)))

        if task_counter == opts.num_tasks:
            break