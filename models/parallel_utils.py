# credit: https://arxiv.org/abs/2311.11077 and https://github.com/adapter-hub/adapters
import torch
from typing import Tuple

def match_attn_matrices_for_parallel(query, key, value) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matches the shapes of query, key and value matrices for parallel composition.
    """
    max_bsz = max(query.shape[0], key.shape[0], value.shape[0])

    query = query.repeat(max_bsz // query.shape[0], *([1] * len(query.shape[1:])))
    key = key.repeat(max_bsz // key.shape[0], *([1] * len(key.shape[1:])))
    value = value.repeat(max_bsz // value.shape[0], *([1] * len(value.shape[1:])))

    return query, key, value
        
def adjust_tensors_for_parallel(hidden_states, *tensors):
    """
    Replicates a given list of tensors based on the shape of the reference tensor (first argument).
    """
    outputs = []
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] >= tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            outputs.append(new_tensor)
        else:
            outputs.append(tensor)
    return tuple(outputs)
        