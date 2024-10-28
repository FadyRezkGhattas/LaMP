from typing import Optional, Any, Dict
from transformers import Trainer
import csv
import os

import torch
import numpy as np
from collections.abc import Mapping

def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()

def nested_concat(tensors, new_tensors):
    assert tensors.shape[1:] == new_tensors.shape[1:]
    C = torch.hstack((tensors, new_tensors))
    return C
    

class EvalLoopContainer:
    """
    Container to store intermediate results of evaluation loop

    Args:
        do_nested_concat (`bool`, *optional*, defaults to `True`):
            If set to `True`, each iteration will recursively concatenate a new object containing tensors to
            the existing stored tensors, provided that the structure of the existing object and the new one
            are identical. If set to `False`, all newly added tensors will be stored in a list.
        padding_index (`int`, *optional*, defaults to -100):
            Value used to pad tensors of different shapes when `do_nested_concat=True`.
    """

    def __init__(self, do_nested_concat: bool = True, padding_index: int = -100):
        self.do_nested_concat = do_nested_concat
        self.padding_index = padding_index
        self.tensors = None
        self.arrays = None

    def add(self, tensors) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        if self.tensors is None:
            self.tensors = tensors if self.do_nested_concat else [tensors]
        elif self.do_nested_concat:
            self.tensors = nested_concat(self.tensors, tensors)
        else:
            self.tensors.append(tensors)

    def to_cpu_and_numpy(self) -> None:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""

        # Check if we have something to add, if not just return
        if self.tensors is None:
            return

        new_arrays = nested_numpify(self.tensors)
        if self.arrays is None:
            self.arrays = new_arrays
        elif self.do_nested_concat:
            self.arrays = nested_concat(self.arrays, new_arrays)
        else:
            self.arrays.extend(new_arrays)

        # reset device tensors after adding to cpu
        self.tensors = None

    def get_arrays(self):
        """Returns the numpified and moved to CPU stored objects."""
        self.to_cpu_and_numpy()
        return self.arrays


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
