import csv
import os
import random
from collections import Counter

import numpy as np
from data import Data

'''
A lot of points per cluster generates noise in the recommendation process since noise tags are considered as research key in the Search Space.
'''

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

def get_query_tags(input_query, task, n_points):
    tag_list    = []
    indexes     = random.sample(range(len(input_query)), n_points)
    input_query = input_query[indexes]
    input_query = list(input_query)

    for item in input_query:
        if task == "img2img" or task == "img2txt":
            query_idx = 1
        elif task == "txt2img" or task == "txt2txt":
            query_idx = 3

        item_tags = item[query_idx].split(',')
        for tag in item_tags:
            # if tag not in tag_list:
            tag_list.append(tag)

    tag_list = [tag for tag, count in sorted(list(Counter(tag_list).items()), key=lambda p:-p[1])[:]]
    sorted_tags           = np.array([tag for tag, count in sorted(list(Counter(tag_list).items()), key=lambda p:-p[1])])
    sorted_counts         = np.array([count for tag, count in sorted(list(Counter(tag_list).items()), key=lambda p:-p[1])])
    threshold             = int(sum(sorted_counts)/3)+1 # at least 33%
    top_n                 = np.argmax(sorted_counts.cumsum()>=threshold)
    tag_list              = sorted_tags[:top_n].tolist()
    return tag_list


def create_pool(pool_size, data, relevants):
    pool        = relevants.tolist()
    indexes     = random.sample(range(len(data.retrieval_data)), pool_size-len(relevants))
    for i in indexes:
        pool.append(data.get_retrieval_item(i))
    random.shuffle(pool)
    return pool


def recommend(query_tags, pool, k, task):
    recommended_items = []

    for item in pool:
        if task == "img2img" or task == "txt2img":
            el_idx = 1
        elif task == "img2txt" or task == "txt2txt":
            el_idx = 3
        item_tags       = set(item[el_idx].split(','))
        # print(item_tags,query_tags)
        n_common_tags = 0
        for a in item_tags:
            for b in set(query_tags):
                if a == b:
                    n_common_tags += 1
        # n_common_tags   = len(item_tags.intersection(set(query_tags)))
        recommended_items.append((item[el_idx-1], n_common_tags))
    recommended_items.sort(key=lambda x:-x[1])

    return recommended_items[:k]

        
if __name__ == '__main__':

    random.seed(42)

    tasks                   = ["txt2img", "img2txt", "img2img", "txt2txt"]
    datasets                = ["coco","flickr30k"]
    tokenizations           = ["_lemma"]
    n_points_per_class      = [5, 10, 20, 30]
    k                       = 10
    
    outpath = 'results/bt_metrics.csv'

    create_path_if_not_existant('results')

    if not os.path.exists(outpath):
        with open(outpath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['task','dataset','tokenization', 'n_seeds', 'n_points_per_class', 'pool size', 'recall','map'])

    for token in tokenizations:

        for dataset in datasets:

            data      = Data(dataset, 90, token)
            n_queries = len(data.query_clusters) 

            for task in tasks:

                for n_seeds in [1,2,5]:

                    n_relevant = n_seeds

                    for n_points in n_points_per_class:

                        for pool_size in [1000]:

                            map     = 0
                            recall  = 0

                            # pick random classes from the first 25 with more samples
                            class_idxs = sample_group(len(data.query_clusters), n_seeds)
                            for c_idxs in class_idxs:
                                classes            = data.query_clusters[c_idxs]
                                queries, relevants = data.get_query_multiple_seed(classes)

                                if task == "img2img" or task == "txt2img":
                                    rel_idx = 0
                                elif task == "img2txt" or task == "txt2txt":
                                    rel_idx = 2
                                
                                relevant_ids         = relevants[:, rel_idx]
                                pool                 = create_pool(pool_size, data, relevants)
                                overall_query_tags   = []
                                for query in queries:
                                    query_tags       = get_query_tags(query, task, n_points)
                                    overall_query_tags.extend(query_tags)
                                
                                # Take the first 33% most occurrent tags
                                sorted_tags           = np.array([tag for tag, count in sorted(list(Counter(overall_query_tags).items()), key=lambda p:-p[1])])
                                sorted_counts         = np.array([count for tag, count in sorted(list(Counter(overall_query_tags).items()), key=lambda p:-p[1])])
                                threshold             = int(sum(sorted_counts)/3)+1 # at least 33%
                                top_n                 = np.argmax(sorted_counts.cumsum()>=threshold)
                                overall_query_tags    = sorted_tags[:top_n].tolist()

                                recommended_items     = recommend(overall_query_tags, pool, k, task)

                                precision = []
                                result    = np.zeros(k)
                                for j, (id, score) in enumerate(recommended_items):
                                    if id in relevant_ids:
                                        result[j] = 1

                                for kk in range(1,k+1):
                                    count = 0
                                    if result[kk-1] == 1:
                                        for l, item in enumerate(result[:kk]):
                                            if item == 1:
                                                count+=1
                                    precision.append(count/kk)

                                map     += (1/n_relevant * np.sum(precision))/len(class_idxs)
                                recall  += (np.sum(result)/n_relevant)/len(class_idxs)

                            row        = [task, dataset, token, n_seeds, n_points, pool_size, round(recall, 2), round(map, 2)]
                            
                            with open(outpath, 'a', encoding='UTF8') as f:
                                writer = csv.writer(f)
                                writer.writerow(row)
