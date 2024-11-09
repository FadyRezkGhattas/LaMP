import torch
from diffusion.net import AdditiveEmbedMLP
from diffusion.transformer import GPT

def get_model(opts):
    if opts.diff_net_arch == 'mlp':
        diffusion_network = AdditiveEmbedMLP(opts.diff_odim, opts.diff_hdim, opts.diff_nhids)
    elif opts.diff_net_arch == 'transformer':
        diffusion_network = GPT({'module_size': opts.lora_rank ** 2, 'indim': opts.diff_odim})
        diffusion_network.indim = opts.diff_odim # required by sampling
    ckpt = torch.load(opts.diff_ckpt, map_location='cuda')
    if "ema" in ckpt:
        diffusion_network.load_state_dict(ckpt["ema"])
    else:
        diffusion_network.load_state_dict(ckpt)
    return diffusion_network