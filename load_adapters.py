import torch
from safetensors import safe_open

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

def load_adapter(model, path, adapter_name='default'):
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

    model.load_state_dict(renamed_state_dict, strict=False)
    return model