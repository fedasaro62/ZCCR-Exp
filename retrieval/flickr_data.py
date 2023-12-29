import random
from typing import Tuple

from torch.utils.data import Dataset

import os
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
import numpy as np

class FlickrCaptions(Dataset):
    """Flickr class that return matching (capt, objects) pairs 50% of times

    Input
    - number_of_samples  : number of sample taken from MSCOCO dataset
    - captions_json      : path to the json containing the images' captions
    - images_path        : path to the folder containg MSCOCO images
    - karpathy           : if the MSCOCO split is the 'karpathy split'

    self.data            : list of samples in the format [imageid, [caption1, caption2, ..., caption n]]
    """
    def __init__(self):
        super(FlickrCaptions, self).__init__()
        instances         = pd.read_csv('/home/mediaverse/ai_hdd/datasets/Flickr30k/results.csv', delimiter='|')[['image_name', ' comment']]
        self.images_path  = '/home/mediaverse/ai_hdd/datasets/Flickr30k/flickr30k_images'
        # with open('/home/mediaverse/Crossmodal-Retrieval-in-Unimodal-and-Multimodal-Search-Spaces/datasets/flickr30k/flickr30k_images/train.txt') as file:
        #     val_image_ids = [line.rstrip()+'.jpg' for line in file][:10000]
        instances         = instances.values # img, caption
        values            = np.array(list(map(lambda x:x[0], instances)))
        idxs              = random.sample(range(len(values)), 10000)
        self.data         = [[x, [y[1] for y in instances if y[0]==x]] for x in tqdm(values[idxs])] # O(N^2) 

    def get_data(self):
        return self.data

    def _get_sample(self, index: int) -> Tuple[str, str]:
        item  = self.data[index]
        n_captions = len(item[1])
        rand = random.randint(0, n_captions-1)
        capt  = item[1][rand]
        img_id = item[0]
        image = Image.open(os.path.join(self.images_path, img_id)).convert('RGB')
        return image, capt

    def samplen(self, n: int):
        return random.sample(self.data, n)

    def __len__(self):
        return len(self.data)
    
    def is_karpathy(self):
        return self.karpathy

    def display_couples(self, n: int) -> None:
        plt.figure(figsize=(22,22))
        col, row = find_couple(n)
        for i in range(n):
            capt, image = self._get_sample(i)
            plt.subplot(row, col, i+1)
            plt.imshow(image)
            plt.title(f"{capt}", color = 'white')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()


def find_couple(n):
    i = 1
    j = 1
    pairs = []
    for i in range(n+1):
        for j in range(n+1):
            if i*j == n:
                pairs.append((i, j))

    pairs = sorted(pairs, key=lambda t: abs(t[0]-t[1]), reverse=False)
    pair = pairs[0]
    if pair[0] > pair[1]:
        col = pair[0]
        row = pair[1]
    else:
        col = pair[1]
        row = pair[0]
    return row, col