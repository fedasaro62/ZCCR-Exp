import re
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from albef.model_pretrain import ALBEF
from albef.tokenization_bert import BertTokenizer
from PIL import Image
from torchvision import transforms


class CustomAlbef:
    def __init__(self, norm=True) -> None:
        config            = yaml.load(open('albef/Pretrain.yaml', 'r'), Loader=yaml.Loader)
        self.text_encoder = 'bert-base-uncased'
        self.tokenizer    = BertTokenizer.from_pretrained(self.text_encoder)
        self.albef        = ALBEF(config=config, tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        self.albef.load_state_dict(torch.load('albef/ALBEF.pth', map_location=torch.device('cpu'))['model'])  # map_location=torch.device('cpu')
        self.albef.eval()
        self.albef.cuda()

        self.name         = 'albef'
        self.norm         = norm

        print("ALBEF - Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.albef.parameters()]):,}")

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

    def pre_caption(self, caption, max_words=30):
        caption = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                caption.lower(),
            )
            .replace("-", " ")
            .replace("/", " ")
        )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])
        return caption

    def encode_text(self, caption: str) -> np.array:
        text                    = self.pre_caption(caption)
        text_input              = self.tokenizer(text, return_tensors="pt")
        text_input              = text_input.to('cuda')
        text_output             = self.albef.text_encoder.bert(text_input.input_ids, attention_mask = text_input.attention_mask,                      
                                                return_dict = True, mode = 'text')            
        text_feat               = text_output.last_hidden_state
        text_feat               = self.albef.text_proj(text_feat[:,0,:])
        if self.norm:
            text_feat               = F.normalize(text_feat, dim=-1)
        embedding               = text_feat.squeeze()#.detach().cpu().numpy().astype(np.float32)
        return embedding
    
    def encode_image(self, rgb_pil_img: Image) -> np.array:
        # image                   = Image.open(img_path).convert('RGB')
        img                     = self.transform(rgb_pil_img).unsqueeze(0)
        img                     = img.cuda()
        image_feat              = self.albef.visual_encoder(img)
        image_feat              = self.albef.vision_proj(image_feat[:,0,:])
        if self.norm:
            image_feat              = F.normalize(image_feat,dim=-1)
        embedding               = image_feat.squeeze()#.detach().cpu().numpy().astype(np.float32)
        return embedding