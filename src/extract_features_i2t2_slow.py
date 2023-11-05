import os
import sys
import pandas as pd
import json
from tqdm import tqdm 
from PIL import Image
import torch
from multiprocessing import Pool
import h5py
from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel

from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT
from third_party.open_clip.clip import tokenize, _transform

import open_clip

logging.set_verbosity_error()

data_dir = 'data/images/'
features_dir = 'i2t_features2/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
annotations = json.load(open('data/dataset_coco.json'))['images']

# org
encoder_name = 'openai/clip-vit-base-patch32'
feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)

# pic2word

img2text = IM2TEXT(embed_dim=768, middle_dim=512, output_dim=768)
state_dict = torch.load('pic2word.pt')["state_dict_img2text"]
for key in list(state_dict.keys()):
    new_key = key.replace("module.", "")
    state_dict[new_key] = state_dict.pop(key)
img2text.load_state_dict(state_dict)

model, _, preprocess = load('ViT-L/14', jit=False)
# tokenize = open_clip.get_tokenizer('ViT-L/14')

img2text = img2text.cuda()
model = model.cuda()


def load_data():
    data = {'train': [], 'val': [], 'test':[]}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'val':
            data['val'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'test':
            data['test'].append({'file_name': file_name, 'cocoid': item['cocoid']})
            
    return data

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    bs = 256
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = df['cocoid'][idx:idx + bs]
        file_names = df['file_name'][idx:idx + bs]
        
        # pic2word feature
        images = torch.stack([preprocess(Image.open(data_dir + file_name).convert("RGB")) for file_name in file_names], dim=0).cuda()
        with torch.no_grad():
            image_features = model.encode_image(images)
            token_features = img2text(image_features.float())
            
            text = tokenize("a photo of").cuda()
            text = text.view(1, -1)
            text = text.repeat(token_features.size(0), 1)
            
            # # 1
            # text_features = model.encode_text_img(text, token_features.half()).unsqueeze(1) # (B, 1, 768)
            
            # # 2
            b_size = token_features.size(0)
            x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]
            collect_ind = text == model.end_id 
            collect_ind = collect_ind.nonzero()[:, 1]
            img_tokens = token_features.half().view(b_size, 1, -1)
            x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            text_features = model.ln_final(x).type(model.dtype)  # (B, 77, 768)
        
        # org features        
        images = [Image.open(data_dir + file_name).convert("RGB") for file_name in file_names]
        with torch.no_grad(): 
            pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(device)
            org_encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state    # (B, 50, 768)
            
        # concat
        encodings = torch.concat((org_encodings, text_features), dim=1).cpu().numpy()
        
        for cocoid, encoding in zip(cocoids, encodings):
            # # 1
            # h5py_file.create_dataset(str(cocoid), (51, 768), data=encoding)
            
            # # # 2
            h5py_file.create_dataset(str(cocoid), (127, 768), data=encoding)
            

data = load_data()

encode_split(data, 'train')
encode_split(data, 'val')
encode_split(data, 'test')
