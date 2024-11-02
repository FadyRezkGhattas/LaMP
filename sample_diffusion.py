import os
import lmdb
import json
import pickle
from argparse import ArgumentParser

import torch
import numpy as np
from utils.utils import mkdir
from data.lmdb import LMDBDataset
from data.lmdb_utils import prepare
from diffusion.net import get_model
from diffusion.sample import greedy_sample
from diffusion.gaussian_diffusion import GaussianDiffusion

parser = ArgumentParser('Inputs: a diffusion model and dataset used to train the diffusion. Sample the diffusion, do necessary transformations and save a model zoo from diffusion prior.')
parser.add_argument('--sampling_routine', default='pcaed', choices=['standard', 'pcaed'], help='If standard, then sample diffusion model, reverse z score using lmdb_addr data if necessary and save diffusion samples in out_dir.\
                    If pcaed, then sample diffusion model. lmdb_addr is assumed to be pcaed data used to train the diff model. The folder must contain pca model, and pre-PCA data statistics.')
parser.add_argument('--lmdb_addr', type=str, default='lmdb_data/LaMP-2-final-pca-1568')
parser.add_argument('--reverse_z_score', type=bool, default=False, help='If True, the lmdb dataset statistics are computed to reverse z-score of model')
parser.add_argument('--diff_ckpt', type=str, default='../grad_free_adapt_diffu/checkpoint/lamp/LaMP-2-final-pca_4x_ln_241030_022536/final_ckpt.pt', help='path to diffusion model for sampling model zoo')
parser.add_argument('--diff_hdim', type=int, default=6272, help='hidden dim of diff net')
parser.add_argument('--diff_nhids', type=int, default=3, help='num of hidden layers in diff net')
parser.add_argument('--diff_odim', type=int, default=1568, help='size of input and output dimensionality of the diffusion model')
parser.add_argument('--diff_sampling_rounds', type=int, default=20, help='Number of rounds to sample diffusion model. Each round, samples 500 datapoints.')
parser.add_argument('--n_worker', type=int, default=8)
parser.add_argument('--out_addr_postfix', type=str, default='')
opts = parser.parse_args()

# load diffusion model
diffusion_net = get_model(opts).to('cuda')
# load diffusion sampler
gaussian_diff = GaussianDiffusion().to('cuda')
# sample model zoo
model_zoo = greedy_sample(gaussian_diff, diffusion_net, opts.diff_sampling_rounds)

if opts.reverse_z_score and opts.sampling_routine == 'standard':
    mean, std = LMDBDataset(opts.lmdb_addr).get_data_stats()
    device = model_zoo[0].device
    model_zoo = [mean.to(device) + (x*std.to(device)) for x in model_zoo]

if opts.sampling_routine == 'pcaed':
    # inverse stats used to train the diffusion model
    mean, std = LMDBDataset(opts.lmdb_addr).get_data_stats()
    device = model_zoo[0].device
    model_zoo = [mean.to(device) + (x*std.to(device)) for x in model_zoo]
    # model zoo are cuda tensors so move them to cpu and cast as np arrays
    model_zoo = [x.cpu().numpy() for x in model_zoo]
    # reverse pca
    with open(os.path.join(opts.lmdb_addr, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    model_zoo = model.inverse_transform(model_zoo)
    # reverse original lmdb stats using saved data
    mean, std = np.load(os.path.join(opts.lmdb_addr, 'means.npy')), np.load(os.path.join(opts.lmdb_addr, 'means.npy'))
    model_zoo = [mean + (x*std) for x in model_zoo]
    # cast zoo back as tensors and
    model_zoo = [torch.from_numpy(x) for x in model_zoo]

out_addr = opts.lmdb_addr+opts.out_addr_postfix+'-diff-samples'
mkdir(out_addr)
with lmdb.open(out_addr, map_size=1024**4, readahead=False) as env:
    prepare(env, model_zoo, opts.n_worker)
with open(os.path.join(out_addr, 'sample_diffusion_args.json'), 'w') as f:
    json.dump(vars(opts), f)