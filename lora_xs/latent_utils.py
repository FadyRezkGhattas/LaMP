import warnings
import torch
import torch.nn.functional as F

import einops

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def get_delta_weight(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    if self.num_adapters > 1:
        raise ValueError('get_delta_weight only supports adapters with a single squared matrix initialized')
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()
    
    latent_name = f'{adapter}_lora_latent_mapping_0'
    output_tensor = transpose(
        weight_B @ getattr(self, latent_name).weight @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor


def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        # if we have multiple adapters loaded and this is the first block, then expand the batch size to repeat it num_adapter times
        if self.num_adapters > 1 and self.first_block:
            x = einops.repeat(x, 'b s d -> (repeat b) s d', repeat=self.num_adapters)

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        latent_name = self.squared_matrix_name
        out = self.lora_dropout[self.active_adapter[0]](x)
        out = self.lora_A[self.active_adapter[0]](out)
        if self.num_adapters > 1:
            shape = out.shape[1:] # first dim is the expanded batch dim -> (batch_size, seq_length, dim)
            out = out.view(self.num_adapters, -1, *shape) # (b, ..) -> (num_adapters, batch_size/num_adapters, seq_length, dim)
            w = torch.transpose(getattr(self, latent_name), 1, 2).unsqueeze(1)
            out = torch.matmul(out, w) # -> (num_adapters, b/num_adapters, seq_length, latent_dim)
            shape = out.shape[2:]
            out = out.view(-1, *shape) # -> (batch_size, seq_length, latent_dim)
        else:
            out = getattr(self, latent_name)(out)
        out = self.lora_B[self.active_adapter[0]](out) # -> (batch_size, seq_length, dim)
        out = out * self.scaling[self.active_adapter[0]]
        result += out
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result