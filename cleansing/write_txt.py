import os

from tqdm import tqdm
from PIL import Image


IMG_DIR = '../../data/Instagram/images'
TXT_DIR = os.path.join(IMG_DIR, '..', 'caption_dataset')
#assert os.path.isdir(IMG_DIR)


def read_txt(txt_path):
    with open(txt_path) as f:
        outs = [line.split(",") for line in f.read().splitlines()]
    return outs


def write_txt(objs, txt_path):
    with open(txt_path, 'w') as f:
        for obj in objs:
            line = ','.join(obj)
            line += '\n'
            f.write(line)
    print(f'text file written: {txt_path}')


def write_sample_txt():
    NPY_DIR = os.path.join('.')

    total_txt = read_txt('train.txt')
    test1_txt = read_txt('test1.txt')
    test2_txt = read_txt('test2.txt')
    total_txt.extend(test1_txt)
    total_txt.extend(test2_txt)

    npy_files = os.listdir(NPY_DIR)

    target_txt = []
    for x in total_txt:
        npy_fn = x[0]
        if npy_fn in npy_files:
            target_txt.append(x)

    write_txt(target_txt, 'trial.txt')


def rm_missing_rows_from_txt(txt_path, new_txt_path):
    outs = read_txt(txt_path)
    
    error, valid_outs = 0, []
    for sample in tqdm(outs):
        npy_fn, _, _, _, _ = sample
        img_path = os.path.join(IMG_DIR, npy_fn[:-4])
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f'[{npy_fn} error: {e}')
            error += 1
            continue
        valid_outs.append(sample)

    write_txt(valid_outs, new_txt_path)

    print(f'total rows: {len(outs)}')
    print(f'total error: {error}')
    return None


if __name__ == '__main__':
    for data_type in ['train']:
        txt_path = os.path.join(TXT_DIR, f'{data_type}.txt')
        new_txt_path = os.path.join(TXT_DIR, f'{data_type}_new.txt')
        rm_missing_rows_from_txt(txt_path, new_txt_path)

