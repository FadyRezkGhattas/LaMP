import lmdb
import pickle
import torch
from tqdm import tqdm
import multiprocessing
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.utils import mkdir
from data.lmdb_utils import prepare
from data.lmdb import LMDBDataset

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='./lmdb_data/LaMP-2-final')
parser.add_argument('--experiment', type=str, default='explained_var', choices=['explained_var', 'dim_red'], help="explained_var creates explained variance plot is saved in data path. \
                    dim_red uses n_components to create a dimensionality reduced dataset and is saved as data name + pca postfix.\
                    For example, ./lmdb_data/LaMP-2-final -> lmdb_data/LaMP-2-final-pca.")
parser.add_argument('--n_components', type=int, default=1568)
parser.add_argument('--out', type=str, default='./lmdb_data/LaMP-2-final-pca')
parser.add_argument('--n_worker', type=int, default=64)
args = parser.parse_args()

print('Loading LMDB Dataset')
model_zoo = LMDBDataset(args.data)
adapters = []
for i in range(len(model_zoo)):
    adapters.append(model_zoo[i].numpy())
adapters = np.array(adapters)

print('Normalizing Data')
means, std = model_zoo.get_data_stats()
adapters = (adapters - means.numpy()) / std.numpy()
adapter_len = adapters[0].shape[0]

if args.experiment == 'explained_var':
    nums = np.arange(16, adapter_len, 16)
    var_ratio=[]
    for n_components in tqdm(nums):
        pca = PCA(n_components=n_components)
        pca.fit_transform(adapters)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))

    # plt.figure(figsize=(4,2),dpi=150)
    plt.grid()
    plt.plot(nums, var_ratio, marker='o')
    plt.xlabel('n_components')
    plt.ylabel('Explained variance ratio')
    plt.title('n_components vs. Explained Variance Ratio')
    plt.savefig(f'{args.data}/pca.jpg')



    for i, j in zip(nums, var_ratio):
        print(i, j)
elif args.experiment == 'dim_red':
    mkdir(args.out)
    # Build model
    print('building model')
    pca = PCA(n_components=args.n_components)
    outs = pca.fit_transform(adapters)
    # Save Model
    print('saving model')
    with open(f'{args.out}/model.pkl','wb') as f:
        pickle.dump(pca,f)
    # Save Adapters Statistics
    print('saving stats')
    np.save(f'{args.out}/means', means)
    np.save(f'{args.out}/std', std)
    # Save dimensionality reduced data
    print('saving dimensionality reduced data')
    with lmdb.open(args.out, map_size=1024**4, readahead=False) as env:
        prepare(env, outs, args.n_worker)
else:
    raise NotImplementedError(f'Given {args.experiment} for --experiment cmd argument is not a valid experiment.')