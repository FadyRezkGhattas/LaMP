import json
from argparse import ArgumentParser

import lmdb
import numpy as np
from data.lmdb import LMDBDataset
from data.lmdb_utils import prepare

parser = ArgumentParser()
parser.add_argument('--data_addr', type=str, default='lmdb_data/LaMP-5-final')
parser.add_argument('--lmdb_clusters', type=str, default='lmdb_data/LaMP-5-final/54_clusters.json')
parser.add_argument('--data_size', type=int, default=4000)
parser.add_argument('--out', type=str, default='lmdb_data/LaMP-5-final-4000')
parser.add_argument('--n_worker', type=str, default=64)
opts = parser.parse_args()

model_zoo = LMDBDataset(opts.data_addr)
with open(opts.lmdb_clusters) as f:
    lmdb_clusters = json.load(f)
medoids_ids = lmdb_clusters['medoids']
clusters_ids = lmdb_clusters['clusters']

num_clusters = len(clusters_ids)
samples_per_cluster = 4000//len(clusters_ids)

# add medoid of each cluster and then random samples from the cluster
subsampled_data = medoids_ids
for i in range(num_clusters):
    cluster_ids = clusters_ids[i]
    cluster_ids.remove(medoids_ids[i])
    if len(cluster_ids) < samples_per_cluster:
        ids = cluster_ids
    else:
        ids = np.random.choice(cluster_ids, samples_per_cluster, replace=False).tolist()
    subsampled_data += ids

adapters = [model_zoo[i] for i in subsampled_data]
with lmdb.open(opts.out, map_size=1024**4, readahead=False) as env:
        prepare(env, adapters, opts.n_worker)