import json

import nltk
import numpy as np
import pandas as pd

nltk.download('stopwords')
from string import punctuation

import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

datasets            = ["coco", "flickr30k"]
stopwords           = stopwords.words('english')
punctuation         = list(punctuation)
nlp                 = spacy.load("en_core_web_sm")
stemmer             = SnowballStemmer(language='english')
removal             = ['ADV', 'ADP', 'CCONJ','DET', 'NUM', 'PART','PRON', 'PROPN','PUNCT', 'SCONJ', 'SPACE', 'SYM']

for dataset in datasets:

    path_data   = "data/"

    df              =   pd.read_csv(path_data+dataset+"_classified.csv", names = ["id","caption","class_id","class_label"])

    with open(path_data+dataset+"_tags.jsonl") as f:
        tags        = [json.loads(line) for line in f]

    # ADD COLUMN WITH IMAGE TAGS
    df["tag_img"] = np.nan

    for item in tags:
        img_id = item["image_id"][:-4]
        if "_" in item["image_visual_contents"]["action"]:
            action_tag = item["image_visual_contents"]["action"].replace("_",",")
        # action_tags_lemma = []
        # action_tags = []
        # for token in nlp(action_tag):
        #     #lemmatization
        #     action_tags_lemma.append(token.lemma_.lower())
        # for w in action_tags_lemma:
        #     action_tags.append(stemmer.stem(w))

        # action_str = ','.join(action_tag)

        
        object_tags = list(set(item["image_visual_contents"]["objects"]))
        # obj_str = ' '.join(object_tags)
        # obj_tags_lemma = []
        # obj_tags = []
        # for token in nlp(obj_str):
        #     obj_tags_lemma.append(token.lemma_.lower())
        # for w in obj_tags_lemma:
        #     obj_tags.append(stemmer.stem(w))

        obj_str = ','.join(object_tags)


        tag_str = obj_str + "," + action_tag

        df.loc[df["id"] == int(img_id), "tag_img"] = tag_str


    # ADD COLUMN WITH TEXT TAGS
    df["tag_txt"] = np.nan

    # remove stop words, punctuation etc from the captions, tokenize, lemmatize and create tags
    for idx, row in df.iterrows():
        caption = row["caption"].lower()

        # spacy
        tag_txt_lemma = []
        tag_txt = []
        
        for token in nlp(caption):
            if token.pos_ not in removal and token.pos_ and not token.is_stop and token.is_alpha:
                tag_txt.append(token.text.lower())
        # for w in tag_txt_lemma:
        #     tag_txt.append(stemmer.stem(w))

        tag_str = ','.join(tag_txt)
        df.loc[idx, "tag_txt"] = tag_str


    df.to_csv(path_data+dataset+"_classified_tags.csv", index = False, header = False)


















