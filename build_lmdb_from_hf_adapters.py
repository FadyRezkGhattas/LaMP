'''
Convert pickle-saved task-wise ckpt files into lmdb.

Each pickle-saved file contains: 
    {'model_vec': model_vec}
'''

import os
import argparse
import multiprocessing
from tqdm import tqdm

import torch
import lmdb
import pickle

from safetensors import safe_open

from utils.utils import mkdir

@torch.no_grad()
def load_pths(pth_file):
    i, file = pth_file
    vector_params = torch.tensor([]).to()
    with safe_open(file, framework="pt", device="cpu") as f:
        for k in f.keys():
            if 'latent_mapping' in k:
                param = f.get_tensor(k)
                vector_param = torch.nn.utils.parameters_to_vector(param)
                vector_params = torch.concat((vector_params.to(vector_param.device), vector_param), dim=0)
    return i, vector_params

def prepare(env, pth_fnames, n_worker):
    files = [(i, file) for i, file in enumerate(pth_fnames)]
    total = 0
    with multiprocessing.Pool(n_worker) as pool:
        for i, params in tqdm(pool.imap_unordered(load_pths, files)):
            key = f'{str(i).zfill(10)}'.encode('utf-8')
            val = pickle.dumps(
                {'model_vec': params}, 
            )
            with env.begin(write=True) as txn:
                txn.put(key, val)
            total += 1
        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--out', type=str, default='lmdb_data/LaMP-2')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--data_path', type=str, default='experiments/LaMP-2/model_zoo/r_6_alpha_16_lr_0.01_epochs_20_sch_linear/ckpts/')

    args = parser.parse_args()
    
    user_folders = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, f))]
    user_adapters = [os.path.join(user_folder, 'adapter_model.safetensors') for user_folder in user_folders]
    mkdir(args.out)
    with lmdb.open(args.out, map_size=1024**4, readahead=False) as env:
        prepare(env, user_adapters, args.n_worker)