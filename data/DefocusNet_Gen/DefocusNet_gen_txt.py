import os
from tqdm import tqdm

path = os.path.abspath('./fs_6')
stack_rgbs = [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith('tif')]
depths = [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith('exr')]

with open('./DefocusNet_train.txt', 'a') as new_train:
    for i in tqdm(range(400)):
        tmp = i*5
        new_train.write('{} {} {} {} {} {}\n'.format(
            stack_rgbs[tmp],
            stack_rgbs[tmp+1],
            stack_rgbs[tmp+2],
            stack_rgbs[tmp+3],
            stack_rgbs[tmp+4],
            depths[i]
            ))

with open('./DefocusNet_val.txt', 'a') as new_train:
    for i in tqdm(range(400, 500)):
        tmp = i*5
        new_train.write('{} {} {} {} {} {}\n'.format(
            stack_rgbs[tmp],
            stack_rgbs[tmp+1],
            stack_rgbs[tmp+2],
            stack_rgbs[tmp+3],
            stack_rgbs[tmp+4],
            depths[i]
            ))
