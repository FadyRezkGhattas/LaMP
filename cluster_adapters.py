import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils.utils import mkdir
from data.lmdb import LMDBDataset

parser = ArgumentParser(description='Clustering lmdb datasets experimets. (1) --experiment=find_num_clusters runs kmeans clustering on the lmdb dataset and saves number of clusters vs silhouette scores plots and data in a subdirectory called clustering_analysis in lmdb data dir.\
                        (2) --experiment=cluster clusters the data into N components and saves a json files called {num_clusters}_clusters.json file in lmdb data dir. The file contains adapter IDs of medoids and IDs of adapters in each cluster.')
parser.add_argument('--data', type=str, default='./lmdb_data/LaMP-2-final')
parser.add_argument('--experiment', type=str, default='find_num_clusters', choices=['cluster', 'find_num_clusters'])
parser.add_argument('--from_num_clusters', type=int, default=2)
parser.add_argument('--to_num_clusters', type=int, default=150)
parser.add_argument('--num_clusters', type=int, default=5)
args = parser.parse_args()

def GetMedoid_id(x, cluster_center):
  return np.argmin([sum((x - cluster_center)**2) for x in x])

model_zoo = LMDBDataset(args.data)
adapters = []
for cluster_id in range(len(model_zoo)):
    adapters.append(model_zoo[cluster_id].numpy())
X_train_norm = preprocessing.normalize(adapters)

if args.experiment == 'find_num_clusters':
    logging_dir = os.path.join(args.data, 'clustering_analysis')
    mkdir(logging_dir)
    fits = []
    score = []
    K = range(args.from_num_clusters, args.to_num_clusters)

    for k in tqdm(K, desc='Num of clusters'):
        # train the model for current value of k on training data
        model = KMeans(n_clusters = k, random_state=0).fit(X_train_norm)
        
        # append the model to fits
        fits.append(model)
        
        # Append the silhouette score to scores
        score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

    plt.grid()
    plt.plot(K, score, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('n_components vs. Explained Variance Ratio')
    plt.savefig(os.path.join(logging_dir, 'clusters_silhouette_score.jpg'))

    df = pd.DataFrame({'k':K,'score':score})
    df.to_csv(os.path.join(logging_dir, 'cluster_scores.csv'), index = False)
elif args.experiment == 'cluster':
    model = KMeans(n_clusters  = args.num_clusters, random_state=0).fit(X_train_norm)
    cluster_centers = model.cluster_centers_
    clusters = []
    clusters_medoids = []
    labels=model.labels_
    for cluster_id in range(args.num_clusters):
        adapter_indices_in_cluster = np.where(labels == cluster_id)[0]
        adapters_in_cluster = np.array(adapters)[adapter_indices_in_cluster]
        medoid_id = GetMedoid_id(adapters_in_cluster, cluster_centers[cluster_id])
        clusters_medoids.append(int(adapter_indices_in_cluster[medoid_id]))
        clusters.append(adapter_indices_in_cluster.tolist())
    with open(f'{args.num_clusters}_clusters.json', 'w') as file:
        json.dump({
            'medoids': clusters_medoids,
            'clusters': clusters
        }, file, indent = 4)