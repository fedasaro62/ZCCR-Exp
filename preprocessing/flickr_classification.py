import argparse
import csv
import os
import random

from classifier import Classifier
from PIL import Image
from tqdm import tqdm


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

def flickr_classified(
        image_root,
        classifier='resnet50',
        capt_data='data/flickr_annotations.csv'):
    
    resnet              = Classifier(classifier)

    savepath = "data/flickr30k_classified.csv"

    with open(capt_data, 'r') as r:
        r.readline() # skip header
        curr_row = 0
        cum      = 0
        for i, line in enumerate(tqdm(r.readlines())):
            if i % 5 == 0:
                # sample a row
                id       = random.sample(range(5), 1)[0]
                row      = cum + id
                curr_row = row
                cum     += 5

            if i == curr_row:
                try:
                    line    = line.split('|')
                    img_id  = line[0].split('.')[0]
                    caption = line[2].strip()
                except:
                    continue

                img_path        = os.path.join(image_root, img_id+'.jpg')
                img             = Image.open(img_path)
                img             = img.convert('RGB')

                # Img Classification
                class_, score, label   = resnet.classify(img)
                if score >= 99.0 and type(caption) == str:
                    with open(savepath, 'a', encoding='UTF8') as w:
                        writer = csv.writer(w)
                        writer.writerow([img_id,caption,str(class_.int().item()),label])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform Flickr classification using a specified classifier.")
    
    parser.add_argument('--image_root', type=str, required=True, help='Root directory containing Flickr images.')
    parser.add_argument('--classifier', type=str, default='resnet50', help='Classifier model to use (default: resnet50).')
    parser.add_argument('--capt_data', type=str, default='data/flickr_annotations.csv', help='Path to the CSV file containing Flickr data.')
    
    args = parser.parse_args()
    
    # Call your function with the parsed arguments
    flickr_classified(args.image_root, args.classifier, args.capt_data)

            
