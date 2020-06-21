import os
import json
from shutil import copyfile

import numpy as np
from PIL import Image
from tqdm import tqdm

REF_JSON_PATH = r'D:\personal\Project\MadeWithML_Incubator\a-PyTorch-Tutorial-to-Image-Captioning\data\caption_datasets\dataset_flickr8k.json'

ROOT = os.path.join('..', '..', 'data', 'instagram')
DATA_DIR = os.path.join(ROOT, 'caption_dataset')
IMG_DIR = os.path.join(ROOT, 'images')
VOCAB_PATH = os.path.join(DATA_DIR, '40000.vocab')
OUT_DIR = os.path.join(os.getcwd(), 'images')


os.makedirs(OUT_DIR, exist_ok = True)
with open(REF_JSON_PATH, 'r') as f:
    ref_json_data = json.load(f)


def read_txt(path):
    with open(path, 'r') as f:
        obj = f.read().splitlines()
    return obj


def decode_one_sample(sample, vocab):
    fn, _, _, token_seq = sample.split(',')
    fn = fn[:-4]
    token_seq = [int(i) for i in token_seq.split('_')]
    word_seq = [vocab[i] for i in token_seq]
    return fn, token_seq, word_seq


def populate_metadata_in_json(data, metadata, data_type = 'test', limit = None):
    # 'val', 'test', 'train'
    assert data_type in ['train', 'val', 'test']

    n = -1 if limit is None else limit
    for i, sample in tqdm(enumerate(data)):
        if i == n:
            break
        
        fn, token_seq, word_seq = decode_one_sample(sample, vocab)

        tmp_sent_dict = dict()
        tmp_sent_dict['tokens'] = word_seq
        tmp_sent_dict['raw'] = ' '.join(word_seq)
        tmp_sent_dict['imgid'], tmp_sent_dict['sentid'] = i, i
        
        tmp_dict = dict()
        tmp_dict['sentids'], tmp_dict['imgid'] = [i], i
        tmp_dict['split'] = data_type
        tmp_dict['filename'] = fn
        tmp_dict['sentences'] = [tmp_sent_dict]
        
        metadata['images'].append(tmp_dict)

    return metadata


vocab = read_txt(VOCAB_PATH)
outs = dict()
outs['dataset'] = 'flickr8k'
outs['images'] = []

for data_file, data_type in [('test1', 'val'), ('test2', 'test'), ('train', 'train')]:
    data_path = os.path.join(DATA_DIR, f'{data_file}.txt')
    data = read_txt(data_path)
    outs = populate_metadata_in_json(data, outs, data_type = data_type, limit = None)


with open('dataset_instagram.json', 'w') as f:
    json.dump(outs, f, indent = 2)