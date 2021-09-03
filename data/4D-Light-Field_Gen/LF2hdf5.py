#####################################################################
# This file is part of the 4D Light Field Benchmark.                #
#                                                                   #
# This work is licensed under the Creative Commons                  #
# Attribution-NonCommercial-ShareAlike 4.0 International License.   #
# To view a copy of this license,                                   #
# visit http://creativecommons.org/licenses/by-nc-sa/4.0/.          #
#####################################################################

import os
import h5py

import argparse

import file_io_py3 as file_io


def convert_to_hdf5(data_folder, disp_all=False):
    """
    input:
        data_folder (list): list of data folders
        disp_all (bool): whether to store high resolution and depth gt
    """
    # init
    scene = dict()
    scene["LF"] = file_io.read_lightfield(data_folder)
    params = file_io.read_parameters(data_folder)

    if params["category"] != "test":
        scene["disp_lowres"] = file_io.read_disparity(data_folder,
                                                      highres=False)
        if disp_all:
            # scene["disp_highres"] = file_io.read_disparity(data_folder,
            #                                                highres=True)
            # scene["depth_highres"] = file_io.read_depth(data_folder,
            #                                             highres=True)
            scene["depth_lowres"] = file_io.read_depth(data_folder,
                                                       highres=False)

    return scene, params


def get_all_data_folders(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    data_folders = []
    data_folders_test = []
    categories = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    for category in categories:
        if category != "test":
            for scene in os.listdir(os.path.join(base_dir, category)):
                data_folder = os.path.join(*[base_dir, category, scene])
                if os.path.isdir(data_folder):
                    data_folders.append(data_folder)
        else:
            for scene in os.listdir(os.path.join(base_dir, category)):
                data_folder = os.path.join(*[base_dir, category, scene])
                if os.path.isdir(data_folder):
                    data_folders_test.append(data_folder)

    # sorted to maintain the same folders
    # reverse to put the addtional at last instead of training
    return sorted(data_folders, reverse=True), sorted(data_folders_test, reverse=True)


if __name__ == '__main__':
    """
    To generate the h5 dataset of the entire Light Field Dataset

    How to run:
        python LF2hdf5.py
            --disp_all (optional)
            --base_dir PATH_TO_ALL_LIGHT_FIELD_DIRS
            --output_dir PATH_TO_SAVE_H5
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--disp_all',
                        action='store_true',
                        help='Whether to save all the disparity and depth')
    parser.add_argument('--base_dir',
                        default='HCI_dataset',
                        help='The base dir to all data')
    parser.add_argument('--output_dir',
                        help='The output dir for Light Field h5 data')
    args = parser.parse_args()

    # init
    print('============ Init h5 File ==================')
    H = 512
    W = 512
    C = 3
    train_h5_path = 'HCI_LF_trainval.h5'
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        train_h5_path = os.path.join(args.output_dir, train_h5_path)

    train_folders, test_folders = get_all_data_folders(args.base_dir)

    # add num val and change num_train
    num_val = 4
    num_train = len(train_folders) - num_val
    num_test = len(test_folders)

    # init h5 file
    dt = h5py.special_dtype(vlen=str)

    if not os.path.exists(train_h5_path):
        with h5py.File(train_h5_path, 'w') as f_train:
            if args.disp_all:
                f_train.create_dataset('disp_train',
                                       shape=(num_train, H, W),
                                       chunks=True)
                # val
                f_train.create_dataset('disp_val',
                                       shape=(num_val, H, W),
                                       chunks=True)
                f_train.create_dataset('depth_train',
                                       shape=(num_train, H, W),
                                       chunks=True)
                # val
                f_train.create_dataset('depth_val',
                                       shape=(num_val, H, W),
                                       chunks=True)
            else:
                f_train.create_dataset('disp_train',
                                       shape=(num_train, H, W),
                                       chunks=True)
                # val
                f_train.create_dataset('disp_val',
                                       shape=(num_val, H, W),
                                       chunks=True)
            f_train.create_dataset('LF_train',
                                   shape=(num_train, 9, 9, H, W, C),
                                   chunks=True)
            f_train.create_dataset('name_train',
                                   shape=(num_train, 1),
                                   dtype=dt,
                                   chunks=True)
            f_train.create_dataset('focal_length_mm_train',
                                   shape=(num_train, 1),
                                   chunks=True)
            f_train.create_dataset('fstop_train',
                                   shape=(num_train, 1),
                                   chunks=True)
            f_train.create_dataset('baseline_mm_train',
                                   shape=(num_train, 1),
                                   chunks=True)
            print(f_train.keys())
            # val
            f_train.create_dataset('LF_val',
                                   shape=(num_val, 9, 9, H, W, C),
                                   chunks=True)
            f_train.create_dataset('name_val',
                                   shape=(num_val, 1),
                                   dtype=dt,
                                   chunks=True)
            f_train.create_dataset('focal_length_mm_val',
                                   shape=(num_val, 1),
                                   chunks=True)
            f_train.create_dataset('fstop_val',
                                   shape=(num_val, 1),
                                   chunks=True)
            f_train.create_dataset('baseline_mm_val',
                                   shape=(num_val, 1),
                                   chunks=True)

    # save data into h5
    for idx in range(num_train):
        # get scenes and params
        print('converting: %s' % train_folders[idx])
        scene, params = convert_to_hdf5(train_folders[idx], args.disp_all)

        # save into h5
        with h5py.File(train_h5_path, 'a') as f_train:
            if args.disp_all:
                f_train['disp_train'][idx] = scene['disp_lowres']
                f_train['depth_train'][idx] = scene['depth_lowres']
            else:
                f_train['disp_train'][idx] = scene['disp_lowres']
            f_train['LF_train'][idx] = scene['LF']
            f_train['name_train'][idx] = train_folders[idx].split('/')[-1]
            f_train['focal_length_mm_train'][idx] = params['focal_length_mm']
            f_train['fstop_train'][idx] = params['fstop']
            f_train['baseline_mm_train'][idx] = params['baseline_mm']
            print("folder name", f_train['name_train'][idx])

    # val
    # save data into h5
    print("Saving Validation Data")
    for idx in range(num_val):
        # get scenes and params
        print('converting: %s' % train_folders[num_train + idx])
        scene, params = convert_to_hdf5(train_folders[num_train + idx], args.disp_all)

        # save into h5
        with h5py.File(train_h5_path, 'a') as f_val:
            if args.disp_all:
                f_val['disp_val'][idx] = scene['disp_lowres']
                f_val['depth_val'][idx] = scene['depth_lowres']
            else:
                f_val['disp_val'][idx] = scene['disp_lowres']
            f_val['LF_val'][idx] = scene['LF']
            f_val['name_val'][idx] = train_folders[num_train + idx].split('/')[-1]
            f_val['focal_length_mm_val'][idx] = params['focal_length_mm']
            f_val['fstop_val'][idx] = params['fstop']
            f_val['baseline_mm_val'][idx] = params['baseline_mm']
            print("folder name", f_val['name_val'][idx])
