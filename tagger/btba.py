import csv
import os
import random
from typing import List

import numpy as np
import torch
from data import Data
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from utils.vanilla_index import Vanilla


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


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

def bert_vectorize(tokenizer, model, text:str):
        encoded_input       = tokenizer(text, return_tensors='pt')
        output              = model(**encoded_input)
        last_hidden_states  = output.last_hidden_state
        return torch.mean(last_hidden_states, 1).squeeze()

def pca_x(x: np.ndarray,n_components):
    x            = PCA(n_components=n_components).fit_transform(x)
    return x

def get_n_components(n_samples:int,n_features:int):
    if min(n_samples,n_features)-1 > 50:
        return 50
    return None

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
    
    # def get_probabilities(self):
    #     return self.clusterer.probabilities_

    def get_n_clusters(self):
        return self.clusterer.n_clusters_

def get_cluster_embeddings(labels:List, n_labels:int):
    '''
     {class_id(str) : emb-idxs(list)}
    '''
    class_embs = {k: [] for k in range(n_labels)}
    for i, label in enumerate(labels):
        if label!=-1:
            class_embs[label].append(i)
    return class_embs
        
if __name__ == '__main__':

    random.seed(42)

    tasks                   = ["txt2img", "img2txt", "img2img", "txt2txt"]
    datasets                = ["coco"] #"flickr30k"
    tokenizations           = ["_lemma"]
    n_points_per_class      = [5, 10, 20, 30]
    k                       = 10
    pool_size               = 1000
    
    outpath = 'results/btba_metrics.csv'

    create_path_if_not_existant('results')

    tokenizer           = BertTokenizer.from_pretrained('bert-base-uncased')
    model               = BertModel.from_pretrained("bert-base-uncased")

    if not os.path.exists(outpath):
        with open(outpath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['task','dataset','tokenization', 'n_seeds', 'n_points_per_class', 'pool size', 'recall','map'])

    for token in tokenizations:

        for dataset in datasets:

            for task in tasks:

                for n_seeds in [1,2,5]:

                    n_relevant = n_seeds

                    for n_points in n_points_per_class:
                        data      = Data(dataset, 90, token, n_points_per_cluster=n_points)
                        n_queries = len(data.query_clusters)

                        if task == "img2img" or task == "txt2img":
                            rel_idx = 0
                        elif task == "img2txt" or task == "txt2txt":
                            rel_idx = 2

                        initial_index  = Vanilla(emb_size=768)
                        # Pool
                        base_idxs      = random.sample(range(len(data.retrieval_data)), pool_size-10)
                        # print(len(relevants), len(idxs))
                        for idx in tqdm(base_idxs):
                            item     = data.get_retrieval_item(idx)
                            tags, id = item[rel_idx+1], item[rel_idx]
                            emb      = bert_vectorize(tokenizer, model, ' '.join(tags.split(',')))
                            # emb     /= emb.norm(dim=-1, keepdim=True)
                            initial_index.add_content(emb.detach().cpu().numpy(), id)
                        
                        initial_index.assets_matrix = initial_index.assets_matrix[1:]

                        map     = 0
                        recall  = 0

                        # pick random classes from the first 25 with more samples
                        class_idxs = sample_group(len(data.query_clusters), n_seeds)
                        for c_idxs in tqdm(class_idxs):
                            index              = initial_index
                            classes            = data.query_clusters[c_idxs]
                            queries, relevants = data.get_query_multiple_seed(classes)
                            
                            relevant_ids         = relevants[:, rel_idx]

                            # # Load relevants
                            for rel, rel_id in zip(relevants[:, rel_idx+1],relevant_ids):
                                emb  = bert_vectorize(tokenizer, model, ' '.join(rel.split(',')))
                                # emb /= emb.norm(dim=-1, keepdim=True)
                                index.add_content(emb.detach().cpu().numpy(), rel_id)
                                print('[UPDATE] -- {}'.format(rel_id))  

                            # # Complete Pool
                            pool_curr = pool_size-10+len(relevant_ids)
                            idxs      = random.sample(range(len(data.retrieval_data)), pool_curr)
                            # print(len(relevants), len(idxs))
                            for idx in tqdm(idxs):
                                if idx not in base_idxs:
                                    if pool_curr == pool_size:
                                        break
                                    item       = data.get_retrieval_item(idx)
                                    tags, id   = item[rel_idx+1], item[rel_idx]
                                    emb        = bert_vectorize(tokenizer, model, ' '.join(tags.split(',')))
                                    # emb       /= emb.norm(dim=-1, keepdim=True)
                                    index.add_content(emb.detach().cpu().numpy(), id)
                                    pool_curr +=1

                            if task == "img2img" or task == "img2txt":
                                query_idx = 1
                            elif task == "txt2img" or task == "txt2txt":
                                query_idx = 3

                            queries    = queries[:,:,query_idx]
                            # Query
                            query_embs = []
                            for clust_ in queries:
                                for tags in clust_:
                                    tags   = ' '.join(tags.split(','))
                                    emb    = bert_vectorize(tokenizer, model, tags)
                                    # emb   /= emb.norm(dim=-1, keepdim=True)
                                    query_embs.append(emb)
                            
                            queries        = torch.vstack(query_embs).detach().cpu().numpy()

                            # PCA
                            n_components   = get_n_components(n_seeds*n_points,768)
                            queries_x      = pca_x(queries,n_components)

                            # Agglomerative Clustering
                            cluster        = Agglomerative(queries_x,5)
                            cluster.fit()
                            class_emb_idxs = get_cluster_embeddings(cluster.get_labels(), cluster.get_n_clusters())

                            retrieved_relevants = []
                            for label in class_emb_idxs:
                                class_emb_idx          = class_emb_idxs[label]
                                query                  = torch.tensor(queries[class_emb_idx])
                                query                  = torch.mean(query, dim=0)
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

                                    map     += (1/n_relevant * np.sum(precision))/len(class_idxs)
                                    recall  += (np.sum(result)/n_relevant)/len(class_idxs)

                        row              = [task, dataset, token, n_seeds, n_points, pool_size, round(recall, 2), round(map, 2)]
                        with open(outpath, 'a', encoding='UTF8') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
