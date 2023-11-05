import json
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Create a custom dataset
class CaptionDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        return self.captions[idx]

class ImageDataset(Dataset):
    def __init__(self, images, image_path, feature_extractor):
        self.images = images
        self.image_path = image_path
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_data = self.images[idx]
        image = self.feature_extractor(Image.open(os.path.join(self.image_path, image_data['file_name'])))
        return image


def load_coco_data(coco_data_path):
    """We load in all images and only the train captions."""

    annotations = json.load(open(coco_data_path))['images']
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
             item['split'] = 'train'
        if item['split'] == 'train':
            for sentence in item['sentences']:
                captions.append({'image_id': item['cocoid'],  'caption': ' '.join(sentence['tokens'])})
        images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})
 
    return images, captions

def filter_captions(data):

    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(caps[idx:idx+bs], return_tensors='np')['input_ids'].tolist()
    
    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions

# 2. Use DataLoader in the encoding functions
def encode_captions(captions, model, device, num_workers=32):

    if os.path.exists('encoded_captions.npy'):
        return np.load('encoded_captions.npy')
    
    dataset = CaptionDataset(captions)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=num_workers)
    
    encoded_captions = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            input_ids = clip.tokenize(batch).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)
    np.save('encoded_captions.npy', encoded_captions)
    
    return encoded_captions


def encode_images(images, image_path, model, feature_extractor, device, num_workers=32):

    image_ids = [i['image_id'] for i in images]

    if os.path.exists('image_features.npy'):
        return image_ids, np.load('image_features.npy')
    
    dataset = ImageDataset(images, image_path, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=num_workers)
    
    image_features = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(batch)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)
    np.save('image_features.npy', image_features)
    
    return image_ids, image_features

def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 

    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions
 
def main(): 

    coco_data_path = 'data/dataset_coco.json' # path to Karpathy splits downloaded from Kaggle
    image_path = 'data/images/'
    
    print('Loading data')
    images, captions = load_coco_data(coco_data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, feature_extractor = clip.load("RN50x64", device=device)

    print('Filtering captions')    
    xb_image_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, device)
    
    print('Encoding images')
    xq_image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)
    
    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)

    print('Writing files')
    faiss.write_index(index, "datastore/coco_index")
    json.dump(captions, open('datastore/coco_index_captions.json', 'w'))
    json.dump(retrieved_caps, open('data/retrieved_caps_resnet50x64.json', 'w'))

if __name__ == '__main__':
    main()