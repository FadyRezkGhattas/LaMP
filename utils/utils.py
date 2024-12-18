from typing import Optional, Any, Dict
from transformers import Trainer
import csv
import os

def mkdir(paths):

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

def opts_to_exp_name(opts):
    return f"r_{opts.rank}_alpha_{opts.lora_alpha}_lr_{opts.learning_rate}_epochs_{opts.num_train_epochs}_sch_{opts.lr_scheduler_type}"


class CSVLogger:
    def __init__(self, output_dir, exp_name):
        self.filepath = os.path.join(output_dir, exp_name.split('/')[-1] + ".csv")

    def log(self, trainer: Optional[Trainer] = None, extra_data: Optional[Dict[str, Any]] = None):
        """Log information about the training process.

        Args:
            trainer (Trainer): The trainer instance.
            extra_data (Optional[Dict[str, Any]], optional): Additional data to log. Defaults to None.
        """
        all_results = {}
        if trainer is not None:
            eval_result = trainer.state.log_history[-2]
            train_result = trainer.evaluate(trainer.train_dataset)
            train_result = {k.replace("eval", "train"): v for k, v in train_result.items()}
            all_results = {**eval_result, **train_result}
        all_results = {**all_results, **extra_data} if extra_data is not None else all_results

        if trainer is not None:
            all_results["train_profile_size"] = len(trainer.train_dataset)
            all_results["eval_profile_size"] = len(trainer.eval_dataset)

        if len(all_results) == 0:
            return
        
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as csvfile:
                w = csv.DictWriter(csvfile, fieldnames=all_results.keys())
                w.writeheader()
        with open(self.filepath, "a", newline="") as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=all_results.keys())
            w.writerow(all_results)
