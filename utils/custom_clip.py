from typing import List

import clip
import numpy as np
import torch
from PIL import Image


class CustomClip:

    def __init__(self, vision_transformer = "ViT-B/32", norm=True):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(vision_transformer)
        self.model.to(DEVICE).eval()

        self.input_resolution = self.model.visual.input_resolution
        self.context_length   = self.model.context_length
        self.vocab_size       = self.model.vocab_size

        self.name             = 'clip'
        self.norm             = norm
        # print('Available models:', clip.available_models())
        print("CLIP - Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        # print("Input resolution:", self.input_resolution)
        # print("Context length:", self.context_length)
        # print("Vocab size:", self.vocab_size)


    def preprocess_image(self, image: Image):
        if image.mode != 'RGB':
            image      = image.convert('RGB')
        return self.preprocess(image)
		
    def encode_text(self, caption: str):
        text_tokens = clip.tokenize(["This is " + desc for desc in [caption]]).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        # if self.norm:
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze()

    # arleady preprocessed images
    def encode_image(self, rgb_pil_image):
        image       = self.preprocess(rgb_pil_image)
        image_input = torch.tensor(np.stack([image])).cuda()
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
        # if self.norm:
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze()