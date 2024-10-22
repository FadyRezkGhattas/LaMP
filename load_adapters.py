import torch
import einops
from safetensors import safe_open

def concatenate_tensorized_adapters(adapters, num_adapters):
    new_adapters = []
    for i in range(0, len(adapters), num_adapters):
        parallel_adapters = adapters[i:i+num_adapters]
        common_keys = set(parallel_adapters[0].keys())
        for item in parallel_adapters:
            common_keys &= set(item.keys())

        # Initialize an empty dictionary to store the concatenated tensors
        result_dict = {}

        # Iterate over the common keys and concatenate the tensors along dim=0
        for key in common_keys:
            tensor_list = [item[key].unsqueeze(0) for item in parallel_adapters]
            result_dict[key] = torch.cat(tensor_list, dim=0)
        new_adapters.append(result_dict)
    return new_adapters

def tensorize_loraxs_adapter(adapter_vector, rank=6, cuda=True, use_bf16=True, adapter_name='default'):
    keys = [
        f'base_model.model.decoder.block.0.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.0.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.0.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.0.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.1.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.1.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.1.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.1.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.10.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.10.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.10.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.10.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.11.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.11.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.11.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.11.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.2.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.2.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.2.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.2.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.3.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.3.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.3.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.3.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.4.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.4.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.4.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.4.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.5.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.5.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.5.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.5.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.6.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.6.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.6.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.6.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.7.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.7.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.7.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.7.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.8.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.8.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.8.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.8.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.9.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.9.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.9.layer.1.EncDecAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.decoder.block.9.layer.1.EncDecAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.0.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.0.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.1.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.1.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.10.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.10.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.11.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.11.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.2.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.2.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.3.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.3.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.4.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.4.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.5.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.5.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.6.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.6.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.7.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.7.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.8.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.8.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.9.layer.0.SelfAttention.q.{adapter_name}_lora_latent_mapping.weight',
        f'base_model.model.encoder.block.9.layer.0.SelfAttention.v.{adapter_name}_lora_latent_mapping.weight',
        ]
    adapter = {}
    start = 0
    i = 0
    while start < len(adapter_vector):
        end = min(start + rank * rank, len(adapter_vector))
        tensor = adapter_vector[start:end].detach().clone().requires_grad_(False)
        tensor = tensor.reshape(rank, rank)
        tensor = tensor.to('cuda').contiguous() if cuda else tensor
        tensor = tensor.bfloat16() if use_bf16 else tensor
        adapter[keys[i]] = tensor
        start += rank * rank
        i += 1
    return adapter

def load_adapter(model, path, adapter_name='default', num_adapters=1):
    tensors = {}
    with safe_open(f"{path}/adapter_model.safetensors",
                   framework="pt",
                   device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    renamed_state_dict = {
            k.replace(
                "lora_A", f"lora_A.{adapter_name}"
            ).replace(
                "lora_B", f"lora_B.{adapter_name}"
            ).replace(
                "_lora_latent", f".{adapter_name}_lora_latent"): v
            for (k, v) in tensors.items() if "classifier.out_proj" not in k
        }
    if num_adapters>1:
        for key in renamed_state_dict:
            if '_lora_latent' in key:
                renamed_state_dict[key] = einops.repeat(renamed_state_dict[key].unsqueeze(0), 'b rank1 rank2 -> (repeat b) rank1 rank2', repeat=num_adapters)
            else:
                continue

    model.load_state_dict(renamed_state_dict, strict=False)
    return model