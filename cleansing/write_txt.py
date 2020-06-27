import os

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