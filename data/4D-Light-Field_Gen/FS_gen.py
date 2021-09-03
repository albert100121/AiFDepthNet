import os

import numpy as np
from tqdm import tqdm, trange
import argparse
import h5py

import subpixel_shift


def focal_stack_generate(LF, stack_size=10, disp_range=[-2.5, 2.5]):
    """
    Args:
        LF (numpy.ndarray):
            the entire light field of one scene with shape 9,9,512,512
        stack_size (int):
            the number for focal stack
        disp_range (list of float):
            the [min, max] value of disparity in focal stack
    output:
        FS
        AiF
    """
    # LF shape, default 9,9,512,512
    x, y, H, W, C = LF.shape

    # init
    lf = LF / 255.0
    uvcenter = np.asarray(np.asarray([x, y]) // 2)  # [4, 4]
    stack = []
    disparities = np.linspace(disp_range[0], disp_range[1], num=stack_size)

    # generating FS
    for idx, disparity in tqdm(enumerate(disparities), desc='disp range'):
        image_cen = np.zeros((H, W, C))

        for u in range(x):
            for v in range(y):
                shift = (uvcenter - np.asarray([u, v])) * disparity

                # center image
                shifted = subpixel_shift.subpixel_shift(
                    np.fft.fft2(np.squeeze(lf[u, v]), axes=(0, 1)), shift, H,
                    W, 1)
                image_cen = image_cen + shifted

        image_cen = image_cen / (x * y)
        image_cen = np.uint8(image_cen * 255.0)

        # create focal stack
        stack.append(image_cen)

        # create AiF
        AiF = lf[int(uvcenter[0]), int(uvcenter[1])] * 255.0

    return np.array(stack), AiF


def main_func(args):
    """
    input:
        args
    output:
        None

    1. init h5py file
    2. load LF, disp, params
    3. Gen h5
    4. store FS, name, focal_length, syn_aperture
    """
    print("=========== Init H5 File ===================")
    if 'train' in args.LF_path:
        args.mode = 'train'
        with h5py.File(args.LF_path, 'r') as dataset:
            num, x, y, H, W, C = dataset['LF_train'].shape

            # add num for val
            if 'val' in args.LF_path:
                num_val = dataset['LF_val'].shape[0]

        num_stack = args.num_stack
        # dt = h5py.string_dtype()
        dt = h5py.special_dtype(vlen=str)

        if not os.path.exists(
                os.path.join(args.output_dir, 'HCI_FS_trainval.h5')):
            print("Dataset does not exist!!!")
            with h5py.File(os.path.join(args.output_dir, 'HCI_FS_trainval.h5'),
                           'w') as f_train:
                f_train.create_dataset('stack_train',
                                       shape=(num, num_stack, H, W, C),
                                       chunks=True)
                f_train.create_dataset('AiF_train',
                                       shape=(num, H, W, C),
                                       chunks=True)
                f_train.create_dataset('disp_train',
                                       shape=(num, H, W),
                                       chunks=True)
                f_train.create_dataset('name_train',
                                       shape=(num, 1),
                                       dtype=dt,
                                       chunks=True)
                f_train.create_dataset('focus_position_disp',
                                       shape=(1, num_stack),
                                       chunks=True)
                f_train.create_dataset('focal_length_mm_train',
                                       shape=(num, 1),
                                       chunks=True)
                f_train.create_dataset('baseline_mm_train',
                                       shape=(num, 1),
                                       chunks=True)
                f_train.create_dataset('syn_aperture_train',
                                       shape=(num, 1),
                                       chunks=True)
                # val
                if 'val' in args.LF_path:
                    f_train.create_dataset('stack_val',
                                           shape=(num_val, num_stack, H, W, C),
                                           chunks=True)
                    f_train.create_dataset('AiF_val',
                                           shape=(num_val, H, W, C),
                                           chunks=True)
                    f_train.create_dataset('disp_val',
                                           shape=(num_val, H, W),
                                           chunks=True)
                    f_train.create_dataset('name_val',
                                           shape=(num_val, 1),
                                           dtype=dt,
                                           chunks=True)
                    f_train.create_dataset('focal_length_mm_val',
                                           shape=(num_val, 1),
                                           chunks=True)
                    f_train.create_dataset('baseline_mm_val',
                                           shape=(num_val, 1),
                                           chunks=True)
                    f_train.create_dataset('syn_aperture_val',
                                           shape=(num_val, 1),
                                           chunks=True)
                print(f_train.keys())
    else:
        args.mode = 'test'
        raise NotImplementedError("Test mode not implemented!!!!!")

    # refocus all
    print("========== Start preparing data ======================")

    disp_range = args.disp_range
    if args.mode == 'train':
        with h5py.File(os.path.join(args.output_dir, 'HCI_FS_trainval.h5'),
                       'a') as f_train:
            f_train['focus_position_disp'][0] = np.linspace(disp_range[0],
                                                            disp_range[1],
                                                            num=num_stack)

    train_idx = val_idx = 0
    # train to train
    for idx in trange(num):
        # load LF, name, disp
        with h5py.File(args.LF_path, 'r') as dataset:
            LF = dataset['LF_%s' % args.mode][idx]
            name = dataset['name_%s' % args.mode][idx, 0].split('/')[-1]
            f_l = dataset['focal_length_mm_%s' % args.mode][idx]
            fstop = dataset['fstop_%s' % args.mode][idx]
            baseline_mm = dataset['baseline_mm_%s' % args.mode][idx]
            if args.mode != 'test':
                disp = dataset['disp_%s' % args.mode][idx]
                # disp = dataset['depth_%s' % args.mode][idx]

        if name not in ['boxes', 'cotton', 'dino', 'sideboard']:
            print(train_idx, name)
            FS, AiF = focal_stack_generate(LF, stack_size=num_stack, disp_range=disp_range)
            with h5py.File(os.path.join(args.output_dir, 'HCI_FS_trainval.h5'),
                           'a') as f_train:
                f_train['stack_train'][train_idx] = FS
                f_train['AiF_train'][train_idx] = AiF
                f_train['disp_train'][train_idx] = disp
                f_train['name_train'][train_idx, 0] = name
                f_train['focal_length_mm_train'][train_idx] = f_l
                f_train['baseline_mm_train'][train_idx] = baseline_mm
                f_train['syn_aperture_train'][train_idx] = 8 * baseline_mm + (
                    f_l / fstop)
            train_idx += 1
    # train to val
    args.mode = 'val'
    for idx in trange(num_val):
        # load LF, name, disp
        with h5py.File(args.LF_path, 'r') as dataset:
            LF = dataset['LF_%s' % args.mode][idx]
            name = dataset['name_%s' % args.mode][idx, 0].split('/')[-1]
            f_l = dataset['focal_length_mm_%s' % args.mode][idx]
            fstop = dataset['fstop_%s' % args.mode][idx]
            baseline_mm = dataset['baseline_mm_%s' % args.mode][idx]
            if args.mode != 'test':
                disp = dataset['disp_%s' % args.mode][idx]

        if name not in ['boxes', 'cotton', 'dino', 'sideboard']:
            print(train_idx, name)
            FS, AiF = focal_stack_generate(LF, stack_size=num_stack, disp_range=disp_range)
            with h5py.File(os.path.join(args.output_dir, 'HCI_FS_trainval.h5'),
                           'a') as f_train:
                f_train['stack_train'][train_idx] = FS
                f_train['AiF_train'][train_idx] = AiF
                f_train['disp_train'][train_idx] = disp
                f_train['name_train'][train_idx, 0] = name
                f_train['focal_length_mm_train'][train_idx] = f_l
                f_train['baseline_mm_train'][train_idx] = baseline_mm
                f_train['syn_aperture_train'][train_idx] = 8 * baseline_mm + (
                    f_l / fstop)
            train_idx += 1

    # train to val
    args.mode = 'train'
    for idx in trange(num):
        # load LF, name, disp
        with h5py.File(args.LF_path, 'r') as dataset:
            LF = dataset['LF_%s' % args.mode][idx]
            name = dataset['name_%s' % args.mode][idx, 0].split('/')[-1]
            f_l = dataset['focal_length_mm_%s' % args.mode][idx]
            fstop = dataset['fstop_%s' % args.mode][idx]
            baseline_mm = dataset['baseline_mm_%s' % args.mode][idx]
            if args.mode != 'test':
                disp = dataset['disp_%s' % args.mode][idx]
                # disp = dataset['depth_%s' % args.mode][idx]

        if name in ['boxes', 'cotton', 'dino', 'sideboard']:
            print(val_idx, name)
            FS, AiF = focal_stack_generate(LF, stack_size=num_stack, disp_range=disp_range)
            with h5py.File(os.path.join(args.output_dir, 'HCI_FS_trainval.h5'),
                           'a') as f_val:
                f_val['stack_val'][val_idx] = FS
                f_val['AiF_val'][val_idx] = AiF
                f_val['disp_val'][val_idx] = disp
                f_val['name_val'][val_idx, 0] = name
                f_val['focal_length_mm_val'][val_idx] = f_l
                f_val['baseline_mm_val'][val_idx] = baseline_mm
                f_val['syn_aperture_val'][val_idx] = 8 * baseline_mm + (f_l /
                                                                        fstop)
            val_idx += 1

    # val to val
    for idx in trange(num_val):
        args.mode = 'val'
        # load LF, name, disp
        with h5py.File(args.LF_path, 'r') as dataset:
            LF = dataset['LF_%s' % args.mode][idx]
            name = dataset['name_%s' % args.mode][idx, 0].split('/')[-1]
            f_l = dataset['focal_length_mm_%s' % args.mode][idx]
            fstop = dataset['fstop_%s' % args.mode][idx]
            baseline_mm = dataset['baseline_mm_%s' % args.mode][idx]
            if args.mode != 'test':
                disp = dataset['disp_%s' % args.mode][idx]

        if name in ['boxes', 'cotton', 'dino', 'sideboard']:
            print(val_idx, name)
            FS, AiF = focal_stack_generate(LF, stack_size=num_stack, disp_range=disp_range)
            with h5py.File(os.path.join(args.output_dir, 'HCI_FS_trainval.h5'),
                           'a') as f_val:
                f_val['stack_val'][val_idx] = FS
                f_val['AiF_val'][val_idx] = AiF
                f_val['disp_val'][val_idx] = disp
                f_val['name_val'][val_idx, 0] = name
                f_val['focal_length_mm_val'][val_idx] = f_l
                f_val['baseline_mm_val'][val_idx] = baseline_mm
                f_val['syn_aperture_val'][val_idx] = 8 * baseline_mm + (f_l /
                                                                        fstop)
            val_idx += 1


if __name__ == '__main__':
    '''
    To Generate Focal-Stack h5py dataset from Light-Field h5py dataset

    input:
        9X9 light field (LF)
    output:
        focal stack (FS)

    how to:
        python --LF_path [path to your light-field h5py]
               --output_dir [path to store your focal-stack h5py]
               --num_stack [the focal-stack's stack size you want to generate]
               --disp_range [The disparity range of your focal-stack]
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--LF_path',
                        default='HCI_LF_train.h5',
                        help='The path to the LF data')
    parser.add_argument(
        '--output_dir',
        default='/media/public_dataset/Defocus_dataset/HCI_FS_DDFF_Dataset',
        help='The output dir path')
    parser.add_argument('--num_stack',
                        default=10,
                        type=int,
                        help='The number of stacked images')
    parser.add_argument('--disp_range',
                        default=[-2.5, 2.5],
                        help='The range of disparity')
    args = parser.parse_args()

    # init
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # main function
    main_func(args)
