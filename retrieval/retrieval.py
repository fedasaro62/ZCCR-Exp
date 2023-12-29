import os
import random
import string
import csv

import numpy as np

from utils.faiss_index import Faiss
from utils.vanilla_index import Vanilla
from utils.custom_clip import CustomClip
from utils.custom_albef import CustomAlbef

from mscoco_data import Captions
from flickr_data import FlickrCaptions

EMB_SIZE = {'clip':512, 'albef':256}

def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

def retrieval_(task             : str,
               dataset          : str,
               model_name       : str,
               outpath          : str,
               n                : int=1000):
    
    assert task in ['txt2img', 'img2txt'], 'Insert txt2img or img2txt as task'
    assert dataset in ['coco', 'flickr30k'], 'Insert coco or flickr30k'
    assert model_name in ['clip', 'albef'], 'Insert clip or albef as model'
    # assert indexer in ['vanilla', 'faiss'], ''
    assert n in [1000, 5000], ''

    if dataset == 'coco':
        captions_json       = '/media/storage/ai_hdd/datasets/MSCOCO/annotations/karpathy_coco_split.json' # captions_val2014.json'
        images_path         = '/media/storage/ai_hdd/datasets/MSCOCO/images'
        data                = Captions(number_of_samples=10000, captions_json=captions_json, images_path=images_path, karpathy=True)

    if dataset == 'flickr30k':
        data                = FlickrCaptions()

    size          = data.__len__()
    emb_size      = EMB_SIZE[model_name]
    model         = CustomClip() if model_name=='clip' else CustomAlbef()
    pool_size     = n
    n_relevant    = 1

    print('SIZE:', size)

    f_index              = Faiss(emb_size=emb_size)
    v_index              = Vanilla(emb_size=emb_size)

    index2id      = {}  # sample index: img_id or txt_id

    # loading data
    counter       = 0
    for idx in random.sample(range(size), 2*pool_size):
        
        img, txt                       = data._get_sample(idx)

        if not isinstance(txt, str):
            continue

        # Texts loading
        if task == 'img2txt':
            emb                  = model.encode_text(txt).squeeze()
            txt_id               = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            f_index.add_content(emb.detach().cpu().numpy(), txt_id)
            v_index.add_content(emb.detach().cpu().numpy(), txt_id)
            print('[UPDATE] -- {}'.format(txt_id))
            index2id[idx]        = txt_id

        # Images loading
        if task == 'txt2img':
            emb             = model.encode_image(img).squeeze()
            img_id          = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            f_index.add_content(emb.detach().cpu().numpy(), img_id)
            v_index.add_content(emb.detach().cpu().numpy(), img_id)
            print('[UPDATE] -- {}'.format(img_id))
            index2id[idx]   = img_id

        counter += 1
        if counter == pool_size:
            break
    
    # print(len(index2id))

    # if indexer == 'vanilla':
    v_index.assets_matrix = v_index.assets_matrix[1:]

    def index_retrieve(index, query,k):
        retrieved_ids, scores = index.retrieve(query, k)
        precision = []
        result    = np.zeros(k)
        for j, id in enumerate(retrieved_ids[:k]):
            if id == relevantid:
                result[j] = 1

        for kk in range(1,k+1):
            count = 0
            if result[kk-1] == 1:
                for l, item in enumerate(result[:kk]):
                    if item == 1:
                        count+=1
            precision.append(count/kk)

        partial_map    = (1/n_relevant * np.sum(precision))/pool_size
        partial_recall = (np.sum(result)/n_relevant)/pool_size
        return partial_recall, partial_map
    
    # QUERIES
    for k in [1,5,10]:
        f_recall, f_map         = 0,0
        v_recall, v_map         = 0,0
        for idx, relevantid in index2id.items():
            img, txt                       = data._get_sample(idx)
            # text query
            if task == 'txt2img':
                emb                         = model.encode_text(txt).squeeze()
                query                       = emb.detach().cpu().numpy()
            # image query
            if task == 'img2txt':
                emb                         = model.encode_image(img).squeeze()
                query                       = emb.detach().cpu().numpy()

            partial_f_recall, partial_f_map = index_retrieve(f_index, query, k)
            partial_v_recall, partial_v_map = index_retrieve(v_index, query, k)
            
            f_recall += partial_f_recall
            f_map    += partial_f_map
            v_recall += partial_v_recall
            v_map    += partial_v_map
            

        f_row    = [task, model_name, dataset, 'faiss', k, pool_size, round(f_recall, 3), round(f_map, 3)]
        v_row    = [task, model_name, dataset, 'vanilla', k, pool_size, round(v_recall, 3), round(v_map, 3)]
        with open(outpath, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(f_row)
            writer.writerow(v_row)

if __name__=='__main__':
    outfile = 'metrics.csv'
    outpath = os.path.join('results', outfile)
    create_path_if_not_existant('results')

    if not os.path.exists(outpath):
        with open(outpath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['task', 'model', 'dataset', 'indexer', 'k', 'pool size', 'recall', 'map'])

    for model in ['clip','albef']:
        for dataset in ['coco', 'flickr30k']:
            for task in ['txt2img', 'img2txt']:
                for pool_size in [1000]:
                    retrieval_(task, dataset, model, outpath, n=pool_size)
    

