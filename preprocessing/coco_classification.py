import argparse
import csv
import json
import os
import random

from classifier import Classifier
from PIL import Image
from tqdm import tqdm


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

def coco_classified(
                    image_root,
                    classifier='resnet50',
                    capt_json='data/karpathy_coco_split.json',):
            
    with open(capt_json, 'r') as fp:
        capt_data = json.load(fp)['images']

    resnet              = Classifier(classifier)

    savepath = "data/coco_classified.csv"

    idxs                = random.sample(range(len(capt_data)), len(capt_data))
    for idx in tqdm(idxs):
        dict            = capt_data[idx]
        img_id          = str(dict['cocoid'])
        sentences       = dict['sentences']

        img_path        = os.path.join(image_root, img_id+'.jpg')
        img             = Image.open(img_path)
        img             = img.convert('RGB')

        # Img Classification
        class_, score, label   = resnet.classify(img)
        if score >= 90.0:
            caption = sentences[random.sample(range(len(sentences)), 1)[0]]['raw']
            if type(caption) != str:
                continue

            with open(savepath, 'a', encoding='UTF8') as w:
                writer = csv.writer(w)
                writer.writerow([img_id,caption,str(class_.int().item()),label])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform COCO classification using a specified classifier.")
    
    parser.add_argument('--image_root',
                        type=str,
                        required=True,
                        help='Root directory containing COCO images.')
    
    parser.add_argument('--classifier',
                        type=str,
                        default='resnet50',
                        help='Classifier model to use (default: resnet50).')
    
    parser.add_argument('--capt_json',
                        type=str,
                        default='data/karpathy_coco_split.json',
                        help='Path to the JSON file containing COCO split information.')
    
    args = parser.parse_args()
    
    coco_classified(args.image_root, args.classifier, args.capt_json)