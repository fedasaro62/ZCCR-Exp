import csv
import os
import random
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm


class Data:
    def __init__(self,
                 dataset:str,
                 n_points_per_cluster:int=30):

        assert dataset in ['coco','flickr30k']

        self.n_points_per_cluster = n_points_per_cluster

        self.dataset_root = 'data'

        self.img_folder   = '{}_images_classified'.format(dataset)

        data = {} # images and text by class
        with open(os.path.join(self.dataset_root, '{}_classified_tags.csv'.format(dataset)), "r") as f:
            reader  = csv.reader(f, delimiter=',')
            counter = 0
            for row in tqdm(reader):
                img_id, caption, label, _ = row
                if label not in data.keys():
                    data[label] = [[img_id,caption]]
                else:
                    data[label].append([img_id,caption])
                counter += 1
                # if counter >= 10000:
                #     break

        clusters                = np.array([id for id, count in
                               sorted([[k, len(data[k])] for k in data],
                                       key=lambda item: -item[1])])
        
        self.query_clusters     = clusters[:25]

        self.retrieval_clusters = clusters[25:]

        self.query_data         = {}
        self.retrieval_data     = []

        for class_ in data:
            if class_ in self.query_clusters:
                self.query_data[class_] = np.array(data[class_])
            else:
                self.retrieval_data.extend(data[class_])

        self.query_data         = self.query_data
        self.retrieval_data     = np.array(self.retrieval_data)

        self.len = counter

    def open_image(self, img_id:str):
        image           = Image.open(os.path.join(self.dataset_root, self.img_folder, '{}.jpg'.format(img_id)))
        image           = image.convert('RGB')
        return image

    def get_query_one_seed(self, class_:str):
        '''
        query list of [rgb pil image, string caption]
        relevant      [rgb pil image, string caption]
        '''
        assert class_ in self.query_clusters

        pair_list = self.query_data[class_]
        # Randomly take 40 as queries to buld the seed and 1 as relevant
        idxs      = random.sample(range(len(pair_list)), self.n_points_per_cluster+1)
        pair_list = pair_list[idxs]
        pair_list = np.array([[self.open_image(img_id), caption] for img_id, caption in pair_list])

        return pair_list[:self.n_points_per_cluster], pair_list[-1]
    
    def get_query_multiple_seed(self, classes:List):
        '''
        query list of [rgb pil image, string caption]
        relevant      [rgb pil image, string caption]
        '''

        query_out = []
        rel_out   = []
        for class_ in classes:
            query, relevant = self.get_query_one_seed(class_)
            query_out.append(query)
            rel_out.append(relevant)

        return np.vstack(query_out), np.vstack(rel_out)


    def get_retrieval_item(self, i):
        '''
        rgb pil image, string caption
        '''
        img_id, caption = self.retrieval_data[i]
        return self.open_image(img_id), caption