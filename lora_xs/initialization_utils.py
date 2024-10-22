import math
import types
from functools import partial

import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch.nn import init
from tqdm import tqdm

from .latent_utils import get_delta_weight, forward_latent
from .svd_utils import get_linear_rec_svd


def get_replacement_module(weight, module_name, type, writer, reconstruct_config):
    cfg = reconstruct_config[type]
    if type == 'svd':
        reconstructed_matrix, enc, dec = get_linear_rec_svd(weight.cpu().detach().numpy(), cfg['rank'],
                                                            cfg['n_iter'],
                                                            cfg['random_state'])
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.tensor):
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2):, :], a=math.sqrt(5))
    return matrix

def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix
  
def set_forward_passes(model, lora_config, adapter_name):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)
            
            target.forward = types.MethodType(forward_latent, target)
            target.get_delta_weight = types.MethodType(get_delta_weight, target)

            latent_name = f'{adapter_name}_lora_latent_mapping'
            setattr(target, latent_name, torch.nn.Linear(lora_config.r, lora_config.r, bias=False))
            getattr(target, latent_name).to(getattr(target.lora_A, adapter_name).weight.device)

            getattr(target.lora_A, adapter_name).weight.requires_grad = False  # only the r*r matrix will be tuned
            getattr(target.lora_B, adapter_name).weight.requires_grad = False  # only the r*r matrix will be tuned

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )

def find_and_initialize(model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer, skip_svd=False, num_adapters=1):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    lora_config = peft_config[adapter_name]
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)
            target.num_adapters = num_adapters
            if 'block.0' in key and 'layer.0' in key:
                target.first_block = True
            else:
                target.first_block = False

            w = target.weight.T
            if not skip_svd:
                replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(weight=w,
                                                                                                    module_name=key,
                                                                                                    type=reconstr_type,
                                                                                                    writer=writer,
                                                                                                    reconstruct_config=reconstruct_config)
            else:
                cfg = reconstruct_config[reconstr_type]
                replacement_encoder_weight = torch.rand([w.shape[0], cfg['rank']], dtype=w.dtype, device=w.device)
                replacement_decoder_weight = torch.rand([cfg['rank'], w.shape[0]], dtype=w.dtype, device=w.device)

            replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
            replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)
            target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
            target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned

            target.forward = types.MethodType(forward_latent, target)
            target.get_delta_weight = types.MethodType(get_delta_weight, target)

            target.squared_matrix_name = 'default_lora_latent_mapping'
            if num_adapters > 1:
                squared_matrix = torch.nn.parameter.Parameter(torch.rand(num_adapters, lora_config.r, lora_config.r))
                setattr(target, target.squared_matrix_name, squared_matrix)
                getattr(target, target.squared_matrix_name).requires_grad = False # if multiple adapters, then this is inference
            else:
                squared_matrix = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
                setattr(target, target.squared_matrix_name, squared_matrix)
                init_module_weights(getattr(target, target.squared_matrix_name), sigma=0.00001)
            
            getattr(target, target.squared_matrix_name).to(target.lora_A.default.weight.device)

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
