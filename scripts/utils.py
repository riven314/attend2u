import os


def read_target_imgs(txt_path):
    with open(txt_path, 'r') as f:
        outs = [line.split(',')[0][:-4] for line in f.read().splitlines()]
    return outs
