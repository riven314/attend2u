import os
import json
from shutil import copyfile

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = os.path.join('..', '..', 'data', 'instagram')
DATA_DIR = os.path.join(ROOT, 'caption_dataset')
IMG_DIR = os.path.join(ROOT, 'images')
VOCAB_PATH = os.path.join(DATA_DIR, '40000.vocab')
DATA_PATH = os.path.join(DATA_DIR, 'test2.txt')
OUT_DIR = os.path.join(os.getcwd(), 'images')

os.makedirs(OUT_DIR, exist_ok = True)


def read_txt(path):
    with open(path, 'r') as f:
        obj = f.read().splitlines()
    return obj


def decode_one_sample(sample, vocab):
    fn, _, _, _, token_seq = sample.split(',')
    fn = fn[:-4]
    token_seq = [int(i) for i in token_seq.split('_')]
    word_seq = [vocab[i] for i in token_seq]
    return fn, token_seq, word_seq


vocab = read_txt(VOCAB_PATH)
data = read_txt(DATA_PATH)
outs = dict()

for i, sample in tqdm(enumerate(data)):
    fn, token_seq, word_seq = decode_one_sample(sample, vocab)
    src = os.path.join(IMG_DIR, fn)

    if not os.path.isfile(src):
        print(f'image not exist: {src}')
        continue
    else:
        tgt = os.path.join(OUT_DIR, fn)
        copyfile(src, tgt)

    out = {'words': word_seq, 'tokens': token_seq, 'filename': fn}
    outs[i] = out


json_out = os.path.join(OUT_DIR, 'test.json')
with open(json_out, 'w') as f:
    json.dump(outs, f)