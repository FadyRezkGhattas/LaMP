from safetensors import safe_open

def load_adapter(model, path):
    tensors = {}
    with safe_open(f"{path}/adapter_model.safetensors",
                   framework="pt",
                   device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    renamed_state_dict = {
            k.replace(
                "lora_A", "lora_A.default"
            ).replace(
                "lora_B", "lora_B.default"
            ).replace(
                "_lora_latent", ".default_lora_latent"): v
            for (k, v) in tensors.items() if "classifier.out_proj" not in k
        }

    model.load_state_dict(renamed_state_dict, strict=False)
    return model