import argparse
import csv
import os

from PIL import Image
from tqdm import tqdm


def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    parser     = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        required=True,
        help='coco or flickr30k',
        type=str
    )

    parser.add_argument(
        '--images_path',
        required=True,
        help='Path to the images directory',
        type=str
    )

    args                  = parser.parse_args()
    dataset               = args.dataset
    img_folder            = args.images_path
    threshold             = 90

    assert dataset in ['coco','flickr30k']

    threshold             = str(threshold)

    out_folder       = os.path.join('data', '{}_images_classified'.format(dataset))
    create_path_if_not_existant(out_folder)

    with open(os.path.join('data', '{}_classified.csv'.format(dataset)), "r") as f:
        reader  = csv.reader(f, delimiter=',')
        counter = 0
        for row in tqdm(reader):
            img_id, _, _, _ = row
            image    = Image.open(os.path.join(img_folder, '{}.jpg'.format(img_id)))
            image    = image.convert('RGB')
            image    = image.save(os.path.join(out_folder, '{}.jpg'.format(img_id)))
            counter += 1

    n_images       = len([entry for entry in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, entry))])
    assert counter == n_images

