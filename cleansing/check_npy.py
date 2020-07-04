"""
check if any npy file specified in [train.txt/ test1.txt/ test2.txt] not exist
"""
import os

from clean import DATA_DIR, read_txt

FEATURE_DIR = '../features'
TXT_PATH = os.path.join(DATA_DIR, 'traingi.txt')

avai_npy_files = os.listdir(FEATURE_DIR)
avai_npy_files = set(avai_npy_files)
partition_data = read_txt(TXT_PATH)

miss_npy_files = []
for sample in partition_data:
    npy_fn, _, _, _, _ = sample.split(',')
    if npy_fn not in avai_npy_files:
        print(f'missing npy: {npy_fn}')
        miss_npy_files.append(sample)

print(f'total: {len(partition_data)}')
print(f'total missing: {len(miss_npy_files)}')
