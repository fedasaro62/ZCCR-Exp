import csv
import os
import random
import string
from typing import List
import numpy as np
import torch
from tqdm import tqdm

import hdbscan

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler


from utils.custom_albef import MyAlbef
from utils.custom_clip import MyClip
from utils.data import Data
from utils.faiss_index import Faiss


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

EMB_SIZE = {'clip':512, 'albef':256}

def GetMedoid(embeddings):
    dist_matrix  = pairwise_distances(embeddings, metric='cosine')
    medoid_index = np.argmin(dist_matrix.sum(axis=0))
    return embeddings[medoid_index]


def sample_group(list_len, k):
    out_idxs  = []
    curr_idxs = []
    counter   = 0
    for idx in random.sample(range(list_len), list_len):
        
        if counter < k:
            curr_idxs.append(idx)
            counter += 1
        
        if counter == k:
            out_idxs.append(curr_idxs)
            curr_idxs = []
            counter   = 0
        
    return out_idxs

def get_n_components(n_samples:int,n_features:int):
    if min(n_samples,n_features)-1 > 50:
        return 50
    return None

def tsne_x(x: np.ndarray, pca:bool, n_components):
    scaler       = StandardScaler()
    x            = scaler.fit_transform(x)

    if pca:
        x        = PCA(n_components=n_components).fit_transform(x)

    x            = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(x)
    
    return x

def pca_x(x: np.ndarray,n_components):
    x            = PCA(n_components=n_components).fit_transform(x)
    return x

class HDBSCAN:
    def __init__(self, embeddings, custer_distance:int, min_cluster_size:int=5):
        self.clusterer = hdbscan.HDBSCAN(alpha=1.0, cluster_selection_epsilon=custer_distance) # min_cluster_size=min_cluster_size
        self.embeddings = embeddings

    def fit(self):
        self.clusterer.fit(self.embeddings)

    def get_labels(self):
        return self.clusterer.labels_
    
    def get_probabilities(self):
        return self.clusterer.probabilities_

    def get_n_clusters(self):
        labels = self.clusterer.labels_
        return len(set(labels)) - (1 if -1 in labels else 0)

class Agglomerative:
    def __init__(self, embeddings,custer_distance:int):
        self.clusterer  = AgglomerativeClustering(n_clusters=None,
                                                  compute_full_tree=True,
                                                  distance_threshold=custer_distance)
        self.embeddings = embeddings

    def fit(self):
        self.clusterer.fit(self.embeddings)

    def get_labels(self):
        return self.clusterer.labels_

    def get_n_clusters(self):
        return self.clusterer.n_clusters_
    
def get_cluster_embeddings(labels:List, n_labels:int):
    '''
     {class_id(str) : emb-idxs(list)}

     # Filter on probability?
    '''
    class_embs = {k: [] for k in range(n_labels)}
    for i, label in enumerate(labels):
        if label!=-1:
            class_embs[label].append(i)
    return class_embs

def retrieval_(task                   : str,
               dataset                : str,
               model_name             : str,
               n_seeds                : int,
               n_points_per_cluster   : int,
               proj                   : str,
               clustering             : str,
               cluster_distance       : int,
               representative         : str,
               outpath                : str,
               n                      : int=1000):
    
    assert task in ['txt2img', 'img2txt', 'txt2txt', 'img2img'], 'Insert txt2img or img2txt as task'
    assert dataset in ['coco', 'flickr30k'], 'Insert coco or flickr30k'
    assert model_name in ['clip', 'albef'], 'Insert clip or albef as model'
    assert n_points_per_cluster in [5,10,20,30], ''
    assert n_seeds in [1,2,5], ''
    assert proj in ['tsne', 'pca', 'None'], ''
    assert clustering in ['hdbscan', 'agglomerative'], ''
    assert cluster_distance in [5,15,25,50], ''
    assert representative in ['centroid', 'medoid'], ''

    data          = Data(dataset,n_points_per_cluster)
    size          = data.len
    emb_size      = EMB_SIZE[model_name]
    model         = MyClip() if model_name=='clip' else MyAlbef()
    pool_size     = n
    n_relevant    = 1

    print('SIZE:', size)

    metrics = {1:{'recall':0, 'map':0},
               5:{'recall':0, 'map':0},
               10:{'recall':0, 'map':0}}

    n_classes  = len(data.query_clusters)
    n_queries  = n_classes

    class_idxs = sample_group(len(data.query_clusters), n_seeds)
    for c_idxs in tqdm(class_idxs):
        classes = data.query_clusters[c_idxs]

        index              = Faiss(emb_size=emb_size)

        query, relevants   = data.get_query_multiple_seed(classes)

        img                = query[:,0]
        txt                = query[:,1]

        img                = torch.vstack([model.encode_image(img).detach().cpu() for img in img]).numpy()
        txt                = torch.vstack([model.encode_text(txt).detach().cpu() for txt in txt]).numpy()

        if task == 'txt2img':
            queries         = txt
            relevant        = relevants[:,0]
        
        if task == 'txt2txt':
            queries         = txt
            relevant        = relevants[:,1]

        if task == 'img2txt':
            queries         = img
            relevant        = relevants[:,1]
        
        if task == 'img2img':
            queries         = img
            relevant        = relevants[:,0]
        
        n_components  = get_n_components(n_seeds*n_points_per_cluster,emb_size)
        if proj=='tsne':                 queries_x = tsne_x(queries, False, n_components)
        elif proj=='pca':                queries_x = pca_x(queries,n_components)
        else:                            queries_x = queries

        # load relevants (one for each cluster, == n_seeds)
        relevant_ids = []
        for rel in relevant:
            if task == 'txt2img' or task == 'img2img': rel            = model.encode_image(rel)
            elif task == 'img2txt' or task == 'txt2txt': rel          = model.encode_text(rel)
            relevant_id  = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            relevant_ids.append(relevant_id)
            index.add_content(rel.detach().cpu().numpy(), relevant_id)
            print('[UPDATE] -- {}'.format(relevant_id))

        # query clustering
        if clustering == 'hdbscan':
            cluster       = HDBSCAN(queries_x,cluster_distance)
        else:
            cluster       = Agglomerative(queries_x,cluster_distance)
        cluster.fit()
        class_emb_idxs    = get_cluster_embeddings(cluster.get_labels(), cluster.get_n_clusters())

        # noisy terms
        counter = n_seeds
        for idx in tqdm(random.sample(range(len(data.retrieval_data)), pool_size)):
    
            img, txt                       = data.get_retrieval_item(idx)
            
            if task == 'txt2img' or task == 'img2img':
                # Image loading
                emb                        = model.encode_image(img).squeeze()
            
            if task == 'img2txt' or task == 'txt2txt':
                # Text loading
                emb                        = model.encode_text(txt).squeeze()

            id               = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            index.add_content(emb.detach().cpu().numpy(), id)

            counter += 1
            if counter == pool_size:
                break
        
        retrieved_relevants = []
        for label in class_emb_idxs:
            class_emb_idx          = class_emb_idxs[label]
            query                  = torch.tensor(queries[class_emb_idx])
            query                  = torch.mean(query, dim=0) if representative=='centroid' else GetMedoid(query)
            query                  = query.detach().cpu().numpy()
            retrieved_ids, scores  = index.retrieve(query, 10)

            # check that a relevant id is retrieved utmost one time for all seeds of a user (it avoids a relevant being counted more times
            # for the same users)
            for k in [10]: # [1,5,10]
                precision = []
                result    = np.zeros(k)
                for j, id in enumerate(retrieved_ids[:k]):
                    if id in relevant_ids and id not in retrieved_relevants:
                        result[j] = 1
                        retrieved_relevants.append(id)

                for kk in range(1,k+1):
                    count = 0
                    if result[kk-1] == 1:
                        for l, item in enumerate(result[:kk]):
                            if item == 1:
                                count+=1
                    precision.append(count/kk)

                metrics[k]['map']     += (1/n_relevant * np.sum(precision))/n_queries
                metrics[k]['recall']  += (np.sum(result)/n_relevant)/n_queries

    for k in [10]:
        recall, map = metrics[k]['recall'], metrics[k]['map']
        row    = [task, model_name, dataset, n_seeds, n_points_per_cluster, proj, clustering,
                  cluster_distance, representative, k, pool_size, round(recall, 2), round(map, 2)]
        with open(outpath, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

if __name__ == '__main__':
    outfile = 'zccr_metrics.csv'
    outpath = os.path.join('results', outfile)
    create_path_if_not_existant('results')

    if not os.path.exists(outpath):
        with open(outpath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['task', 'model', 'dataset', 'n_seeds', 'n_points_per_cluster', 'proj', 'clustering',
                             'cluster distance', 'representative', 'k', 'pool size', 'recall', 'map'])

    for model in ['clip','albef']:  # clip
        for dataset in ['coco', 'flickr30k']: #'flickr30k'
            for task in ['txt2img', 'txt2txt', 'img2img', 'img2txt']: #'txt2img'
                for proj in ['pca']: # 'kernel pca rbf', 'kernel pca cosine', 'tsne (pca)', 'tsne', 'pca', 'None'
                    for clustering in ['hdbscan', 'agglomerative']:
                        for cluster_distance in [5]: #15,25,50
                            for representative in ['centroid', 'medoid']:
                                for n_points_per_cluster in [5,10,20,30]:
                                    for n_seeds in [1,2,5]:
                                        retrieval_(task, dataset, model, n_seeds, n_points_per_cluster,
                                                    proj, clustering, cluster_distance, representative, outpath)
