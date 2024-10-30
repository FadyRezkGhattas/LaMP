import pickle
import torch
from tqdm import tqdm
import multiprocessing

@torch.no_grad()
def load_data(data):
    i, params = data
    vector_params = torch.tensor(params)
    return i, vector_params

def prepare(env, data, n_worker):
    total = 0
    data = [(i, file) for i, file in enumerate(data)]
    with multiprocessing.Pool(n_worker) as pool:
        for i, params in tqdm(pool.imap_unordered(load_data, data)):
            key = f'{str(i).zfill(10)}'.encode('utf-8')
            val = pickle.dumps(
                {'model_vec': params}, 
            )
            with env.begin(write=True) as txn:
                txn.put(key, val)
            total += 1
        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))