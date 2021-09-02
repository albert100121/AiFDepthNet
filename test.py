import os
import argparse
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
import skimage.filters as skf
# for output results
from imageio import imread, imwrite
from matplotlib import cm

import utils

# Arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--txt', help='path to image name txt file')
parser.add_argument('--h5py', help='path to image name txt file')
parser.add_argument('--pth', help='path to dumped .pth file', required=True)
parser.add_argument('--outdir', required=False)
parser.add_argument('--dataset', default='DefocusNet')  # DefocusNet, DDFF, ...
parser.add_argument('--disp_depth', default='depth')  # DefocusNet, DDFF, ...
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

# force dataset type upper character
args.dataset = args.dataset.upper()
# check if path entered
if not args.txt and not args.h5py:
    print("DATASET PATH MISSING!!!")


def DefocusNet_testing(args):
    # Prepare all input rgb paths
    # load data
    if args.txt is not None:
        rgb1_paths = np.empty(0)
        rgb2_paths = np.empty(0)
        rgb3_paths = np.empty(0)
        rgb4_paths = np.empty(0)
        rgb5_paths = np.empty(0)
        depth_paths = np.empty(0)
        with open(args.txt, 'r') as f:
            for line in tqdm(f.readlines(), desc='Load paths'):
                tmp = line.strip().split()
                rgb1_paths = np.append(rgb1_paths, tmp[0])
                rgb2_paths = np.append(rgb2_paths, tmp[1])
                rgb3_paths = np.append(rgb3_paths, tmp[2])
                rgb4_paths = np.append(rgb4_paths, tmp[3])
                rgb5_paths = np.append(rgb5_paths, tmp[4])
                depth_paths = np.append(depth_paths, tmp[5])

    for path in rgb1_paths:
        assert os.path.isfile(path) or os.path.islink(path)

    length = len(depth_paths)
    print('%d images in total.' % length)

    # Load trained checkpoint
    print('Loading checkpoint...', end='', flush=True)
    net, args_dict, args_model = utils.load_trained_model(args.pth)
    net = net.eval().to(args_dict['device'])
    for k, v in args_dict.items():
        if k not in args:
            setattr(args, k, v)
    print('done')

    # create output path
    if args.outdir:
        dirs = ['pred_depth', 'pred_AiF', 'pred_depth_jpg']
        outpath = os.path.join(args.outdir, args.dataset)
        for d in dirs:
            os.makedirs(os.path.join(outpath, d), exist_ok=True)

    losses = dict()
    losses['MAE'] = 0
    losses['MSE'] = 0
    losses['RMSE'] = 0
    losses['MAE_2'] = 0
    losses['MSE_2'] = 0
    losses['RMSE_2'] = 0
    losses['MAE_fp'] = 0
    losses['MSE_fp'] = 0
    losses['RMSE_fp'] = 0
    losses['logRMSE_fp'] = 0
    losses['absRel_fp'] = 0
    losses['sqrRel_fp'] = 0
    losses['time'] = 0
    L1loss_fn = nn.L1Loss()
    L2loss_fn = nn.MSELoss()
    focus_position = [0.1, 0.15, 0.3, 0.7, 1.5]

    for idx in tqdm(range(length)):
        k = os.path.split(depth_paths[idx])[-1][:-4]

        with torch.no_grad():
            # get the gt and mask
            gt = torch.from_numpy(
                imread(depth_paths[idx])[..., 0][None, None].astype(
                    np.float32)).to(args.device)
            mask = gt > 0
            mask_2 = (gt > 0) & (gt <= 2)
            mask_fp = (gt >= min(focus_position)) & (gt <= max(focus_position))

            input_dict = {
                'stack_rgb_img':
                torch.cat([
                    torch.from_numpy(
                        imread(rgb1_paths[idx]).transpose(
                            2, 0, 1)[None].astype(np.float32)).to(
                                args.device).unsqueeze(2) / 255,
                    torch.from_numpy(
                        imread(rgb2_paths[idx]).transpose(
                            2, 0, 1)[None].astype(np.float32)).to(
                                args.device).unsqueeze(2) / 255,
                    torch.from_numpy(
                        imread(rgb3_paths[idx]).transpose(
                            2, 0, 1)[None].astype(np.float32)).to(
                                args.device).unsqueeze(2) / 255,
                    torch.from_numpy(
                        imread(rgb4_paths[idx]).transpose(
                            2, 0, 1)[None].astype(np.float32)).to(
                                args.device).unsqueeze(2) / 255,
                    torch.from_numpy(
                        imread(rgb5_paths[idx]).transpose(
                            2, 0, 1)[None].astype(np.float32)).to(
                                args.device).unsqueeze(2) / 255
                ], 2),
                'focus_position':
                torch.tensor([0.1, 0.15, 0.3, 0.7,
                              1.5]).view(1, 1, 5).to(args.device)
            }

            # Call network for inference
            start_time = time.time()
            infer_dict = net.module.inference(input_dict, args)
            losses['time'] += (time.time() - start_time) / length

            # calculate loss
            pred_depth = infer_dict['pred_{}'.format(args.disp_depth)]
            # depth >0
            losses['MAE'] += L1loss_fn(pred_depth[mask], gt[mask]) / length
            losses['MSE'] += L2loss_fn(pred_depth[mask], gt[mask]) / length
            losses['RMSE'] += (L2loss_fn(pred_depth[mask], gt[mask]) **
                               0.5) / length
            # 0 < depth <= 2 meters
            losses['MAE_2'] += L1loss_fn(pred_depth[mask_2],
                                         gt[mask_2]) / length
            losses['MSE_2'] += L2loss_fn(pred_depth[mask_2],
                                         gt[mask_2]) / length
            losses['RMSE_2'] += (L2loss_fn(pred_depth[mask_2], gt[mask_2]) **
                                 0.5) / length
            # focus_min <= depth <= focus_max
            losses['MAE_fp'] += L1loss_fn(pred_depth[mask_fp],
                                          gt[mask_fp]) / length
            losses['MSE_fp'] += L2loss_fn(pred_depth[mask_fp],
                                          gt[mask_fp]) / length
            losses['RMSE_fp'] += (L2loss_fn(pred_depth[mask_fp], gt[mask_fp]) **
                                  0.5) / length
            losses['logRMSE_fp'] += (L2loss_fn(torch.log(
                pred_depth[mask_fp]), torch.log(gt[mask_fp]))**0.5) / length
            losses['absRel_fp'] += torch.mean(
                abs(pred_depth[mask_fp] - gt[mask_fp]) / gt[mask_fp])
            losses['sqrRel_fp'] += torch.mean(
                ((pred_depth[mask_fp] - gt[mask_fp])**2) / gt[mask_fp])

        # Dump results
        if args.outdir:
            # write exr
            imwrite(os.path.join(outpath, 'pred_depth', k + '.exr'),
                    pred_depth.cpu()[0, 0])
            # write jpg in jet
            cmap = cm.get_cmap('jet')
            # normalized jet visualization
            color_img = cmap(
                ((torch.clamp(pred_depth, min(focus_position),
                              max(focus_position)) - min(focus_position)) /
                    (max(focus_position) - min(focus_position))).cpu().numpy()[0, 0])[..., :3]
            # write results
            imwrite(os.path.join(outpath, 'pred_depth_jpg', k + '.jpg'),
                    color_img,
                    quality=100)
            imwrite(os.path.join(outpath, 'pred_AiF', k + '.jpg'),
                    infer_dict['pred_AiF_img'][0].cpu().numpy().transpose(
                        1, 2, 0),
                    quality=100)
    return losses


def DDFF_testing(args):

    # assertion
    assert os.path.isfile(args.h5py)

    # init dataset
    with h5py.File(args.h5py, 'r') as dataset:
        if args.test:
            stack_rgb_img = dataset['stack_test'][:] / 255.
        else:
            stack_rgb_img = dataset['stack_val'][:] / 255.
            disp = dataset['disp_val'][:]

    length, S, H, W, C = stack_rgb_img.shape

    # focus position
    focal_length = 521.4052
    K2 = 1982.0250823695178
    flens = 7317.020641763665
    baseline = K2 / flens * 1e-3
    focus_position = torch.from_numpy(
        np.linspace(baseline * focal_length / 0.5,
                    baseline * focal_length / 7,
                    num=S)[None])
    # padding
    pad_h = pad_w = 0
    if H % 32 != 0:
        pad_h = 32 - (H % 32)
        stack_rgb_img = np.pad(stack_rgb_img,
                               ((0, 0), (0, 0), (0, pad_h), (0, 0), (0, 0)),
                               mode='constant',
                               constant_values=(0, 0))
    if W % 32 != 0:
        pad_w = 32 - (W % 32)
        stack_rgb_img = np.pad(stack_rgb_img,
                               ((0, 0), (0, 0), (0, 0), (0, pad_w), (0, 0)),
                               mode='constant',
                               constant_values=(0, 0))

    # Load trained checkpoint
    print('Loading checkpoint...', end='', flush=True)
    net, args_dict, args_model = utils.load_trained_model(args.pth)
    net = net.eval().to(args_dict['device'])
    for k, v in args_dict.items():
        if k not in args:
            setattr(args, k, v)
    print('done')

    # create output path
    if args.outdir:
        if args.test:
            outpath = os.path.join(args.outdir, args.dataset, 'test')
        else:
            outpath = os.path.join(args.outdir, args.dataset, 'val')
        os.makedirs(outpath, exist_ok=True)

    losses = dict()
    losses['MAE'] = 0
    losses['MSE'] = 0
    losses['RMSE'] = 0
    losses['MAE_fp'] = 0
    losses['MSE_fp'] = 0
    losses['RMSE_fp'] = 0
    losses['time'] = 0
    L1loss_fn = nn.L1Loss()
    L2loss_fn = nn.MSELoss()

    for idx in tqdm(range(length)):
        k = str(idx)

        with torch.no_grad():
            if not args.test:
                # get the gt and mask
                gt = torch.from_numpy(disp[idx][None, None].astype(
                    np.float32)).to(args.device)
                mask = gt > 0
                # mask_2 = (gt > 0) & (gt <= 2)
                mask_fp = (gt >= torch.min(focus_position)) & (
                    gt <= torch.max(focus_position))

            input_dict = {
                'stack_rgb_img':
                torch.from_numpy(stack_rgb_img[idx].transpose(
                    3, 0, 1, 2)[None].astype(np.float32)).to(args.device),
                'focus_position':
                focus_position.to(args.device)
            }

            # Call network for inference
            start_time = time.time()
            infer_dict = net.module.inference(input_dict, args)
            runtime = time.time() - start_time
            losses['time'] += runtime / length

            # calculate loss
            pred_disp = infer_dict['pred_disp']
            if pad_h:
                pred_disp = pred_disp[..., :-pad_h, :]
            if pad_w:
                pred_disp = pred_disp[..., :-pad_w]
            if not args.test:
                losses['MAE'] += L1loss_fn(pred_disp[mask], gt[mask]) / length
                losses['MSE'] += L2loss_fn(pred_disp[mask], gt[mask]) / length
                losses['RMSE'] += (L2loss_fn(pred_disp[mask], gt[mask]) **
                                   0.5) / length
                losses['MAE_fp'] += L1loss_fn(pred_disp[mask_fp],
                                              gt[mask_fp]) / length
                losses['MSE_fp'] += L2loss_fn(pred_disp[mask_fp],
                                              gt[mask_fp]) / length
                losses['RMSE_fp'] += (L2loss_fn(pred_disp[mask_fp],
                                                gt[mask_fp])**0.5) / length

        # Dump results
        if args.outdir:
            if args.test:
                # get scene name
                if args.dataset == 'DDFF':
                    scenes = [
                        'lockeroom', 'cafeteria', 'library', 'spencerlab',
                        'office44', 'magistrale'
                    ]
                # get scene idx and image idx, generate names for path
                scene_idx, img_idx = divmod(idx, 20)
                img_idx = str(img_idx + 1).zfill(4)
                scene = scenes[scene_idx]
                # create dir for numpy files
                tmp_path = os.path.join(outpath, 'numpy', scene)
                name = 'DISP_' + img_idx + '.npy'
                os.makedirs(tmp_path, exist_ok=True)
                # save numpy data
                np.save(os.path.join(tmp_path, name), pred_disp.cpu()[0, 0])
                with open(os.path.join(outpath, 'numpy', 'runtime.txt'),
                          'a') as f:
                    f.write("{}/{} {}\n".format(scene, name[:-4],
                                                str(runtime)))
                tmp_path = os.path.join(outpath, 'jpg', scene)
                name = 'DISP_' + k + '.jpg'
                os.makedirs(tmp_path, exist_ok=True)
                # convert images to jet and save images
                cmap = cm.get_cmap('jet')
                color_img = cmap(
                    ((pred_disp - torch.min(focus_position)) / (torch.max(focus_position) - torch.min(focus_position))).cpu().numpy()[0, 0])[..., :3]
                imwrite(os.path.join(tmp_path, name), color_img[:272, :416], quality=100)
                # save AiF
                # create dir for jpg files
                tmp_path = os.path.join(outpath, 'AiF_jpg', scene)
                name = 'AiF_' + k + '.jpg'
                os.makedirs(tmp_path, exist_ok=True)
                color_img = torch.cat([infer_dict['pred_AiF_img'][:, 2], infer_dict['pred_AiF_img'][:, 1], infer_dict['pred_AiF_img'][:, 0]], 0)
                imwrite(os.path.join(tmp_path, name), color_img.cpu().numpy().transpose(1, 2, 0)[:272, :416])
            else:
                # create dir for depth exr files
                tmp_path = os.path.join(outpath, 'exr')
                name = 'DISP_' + k + '.exr'
                os.makedirs(tmp_path, exist_ok=True)
                # convert images to jet and save images
                imwrite(os.path.join(tmp_path, name), pred_disp.cpu()[0, 0])

                # create dir for depth jpg files
                tmp_path = os.path.join(outpath, 'jet_jpg')
                name = 'DISP_' + k + '.jpg'
                os.makedirs(tmp_path, exist_ok=True)
                # convert images to jet and save images
                cmap = cm.get_cmap('jet')
                pred_disp_viz = (torch.clamp(pred_disp, focus_position.min(), focus_position.max()) / focus_position.max()).cpu().numpy()[0, 0]
                color_img = cmap(pred_disp_viz)[..., :3]
                imwrite(os.path.join(tmp_path, name), color_img)

                # create dir for AiF jpg files
                tmp_path = os.path.join(outpath, 'AiF_jpg')
                name = 'AiF_' + k + '.jpg'
                os.makedirs(tmp_path, exist_ok=True)
                color_img = torch.cat([infer_dict['pred_AiF_img'][:, 2], infer_dict['pred_AiF_img'][:, 1], infer_dict['pred_AiF_img'][:, 0]], 0)
                imwrite(os.path.join(tmp_path, name), color_img.cpu().numpy().transpose(1, 2, 0))

    return losses


def Mobile_Depth_testing(args):
    # Prepare all input rgb paths
    # load data
    stack_rgbs = []
    FPs = []
    if args.txt is not None:
        with open(args.txt, 'r') as f:
            for line in tqdm(f.readlines(), desc='Load paths'):
                tmp = line.strip().split()
                stack_rgbs.append(tmp)
        with open('data/Mobile_FP.txt', 'r') as f:
            for line in tqdm(f.readlines(), desc='Load FP'):
                tmp = list(map(float, line.strip().split()))
                FPs.append(tmp)

    length = len(stack_rgbs)
    print('%d images in total.' % length)

    # Load trained checkpoint
    print('Loading checkpoint...', end='', flush=True)
    net, args_dict, args_model = utils.load_trained_model(args.pth)
    net = net.eval().to(args_dict['device'])
    for k, v in args_dict.items():
        if k not in args:
            setattr(args, k, v)
    print('done')

    # create output path
    if args.outdir:
        if 'Aligned' not in args.txt:
            args.dataset += '_ORI'
        dirs = ['pred_depth', 'pred_AiF', 'pred_depth_jet']
        args.stack_num = 10
        tmp = str(args.stack_num)
        outpath = os.path.join(args.outdir, args.dataset, tmp)
        for d in dirs:
            os.makedirs(os.path.join(outpath, d), exist_ok=True)

    losses = dict()
    losses['time'] = 0

    for idx in tqdm(range(length)):
        k = os.path.split(stack_rgbs[idx][0])[0].split('/')[-1]

        with torch.no_grad():

            # sort by depth
            FP = FPs[idx]
            if k == 'metals':
                FP = FP[::-1]
            rgb_paths = stack_rgbs[idx]
            rgb_FP = []
            for i in range(len(FP)):
                rgb_FP.append([FP[i], rgb_paths[i]])
            rgb_FP.sort(key=lambda x: x[0])
            rgb_paths = [x[1] for x in rgb_FP]
            FP = [x[0] for x in rgb_FP]

            # original FP or normalized FP
            if args.stack_num:
                print("========= stack num {} ===============".format(
                    args.stack_num))
                step = max(len(FP) // args.stack_num, 1)
                rgb_paths = rgb_paths[::step]
                FP = (np.arange(1, len(FP) + 1) / len(FP))[::step]

            # S, H, W, C => C, S, H, W
            stack_rgb_img = np.array([imread(x) for x in rgb_paths]).astype(
                np.float32).transpose(3, 0, 1, 2)
            C, S, H, W = stack_rgb_img.shape
            # padding
            pad_h = pad_w = 0
            if H % 32 != 0:
                pad_h = 32 - (H % 32)
                stack_rgb_img = np.pad(stack_rgb_img,
                                       ((0, 0), (0, 0), (0, pad_h), (0, 0)))
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
                stack_rgb_img = np.pad(stack_rgb_img,
                                       ((0, 0), (0, 0), (0, 0), (0, pad_w)))
            stack_rgb_img = torch.from_numpy(stack_rgb_img[None]).to(
                args.device) / 255.

            input_dict = {
                'stack_rgb_img':
                stack_rgb_img,
                'focus_position':
                torch.tensor(FP).view(1, 1, S).to(args.device)
            }

            # Call network for inference
            start_time = time.time()
            infer_dict = net.module.inference(input_dict, args)
            losses['time'] += (time.time() - start_time) / length

            # calculate loss
            pred_depth = infer_dict['pred_{}'.format(args.disp_depth)]
            pred_AiF = infer_dict['pred_AiF_img']
            if pad_h:
                pred_depth = pred_depth[..., :-pad_h, :]
                pred_AiF = pred_AiF[..., :-pad_h, :]
            if pad_w:
                pred_depth = pred_depth[..., :-pad_w]
                pred_AiF = pred_AiF[..., :-pad_w]

        # Dump results
        if args.outdir:
            imwrite(os.path.join(outpath, 'pred_depth', k + '.exr'),
                    pred_depth.cpu()[0, 0])
            cmap = cm.get_cmap('jet')
            color_img = cmap(pred_depth.cpu().numpy()[0, 0])[..., :3]
            imwrite(
                os.path.join(outpath, 'pred_depth_jet', k + '.jpg'),
                color_img, quality=100)
            imwrite(os.path.join(outpath, 'pred_AiF', k + '.jpg'),
                    pred_AiF[0].cpu().numpy().transpose(1, 2, 0), quality=100)
    return losses


def HCI_testing(args):

    # assertion
    assert os.path.isfile(args.h5py)

    # init dataset
    with h5py.File(args.h5py, 'r') as dataset:
        stack_rgb_img = dataset['stack_val'][:] / 255.
        disp = dataset['disp_val'][:]
        name = dataset['name_val'][:]
        focus_position = dataset['focus_position_disp'][:]
    length, S, H, W, C = stack_rgb_img.shape

    # focus position
    focus_position = torch.from_numpy(focus_position)

    # Load trained checkpoint
    print('Loading checkpoint...', end='', flush=True)
    net, args_dict, args_model = utils.load_trained_model(args.pth)
    net = net.eval().to(args_dict['device'])
    for k, v in args_dict.items():
        if k not in args:
            setattr(args, k, v)
    print('done')

    # create output path
    if args.outdir:
        outpath = os.path.join(args.outdir, args.dataset)
        os.makedirs(outpath, exist_ok=True)

    # focus position
    input_dict = {'focus_position': focus_position.to(args.device)}

    # padding
    pad_h = pad_w = 0
    if H % 32 != 0:
        pad_h = 32 - (H % 32)
        stack_rgb_img = np.pad(stack_rgb_img, (0, 0), (0, 0), (0, pad_h),
                               (0, 0), (0, 0))
    if W % 32 != 0:
        pad_w = 32 - (W % 32)
        stack_rgb_img = np.pad(stack_rgb_img, (0, 0), (0, 0), (0, 0),
                               (0, pad_w), (0, 0))

    # create losses dict
    losses = dict()
    losses['MAE'] = 0
    losses['MSE'] = 0
    losses['RMSE'] = 0
    losses['MAE_fp'] = 0
    losses['MSE_fp'] = 0
    losses['RMSE_fp'] = 0
    losses['logRMSE_fp'] = 0
    losses['absRel_fp'] = 0
    losses['sqrRel_fp'] = 0
    losses['Badpixel0.07'] = 0
    losses['Bumpiness'] = 0
    losses['time'] = 0
    L1loss_fn = nn.L1Loss()
    L2loss_fn = nn.MSELoss()

    def get_bumpiness(gt, algo_result, mask, clip=0.05, factor=100):
        # init
        # import skimage.filters as skf
        if type(gt) == torch.Tensor:
            gt = gt.cpu().numpy()[0, 0]
        if type(algo_result) == torch.Tensor:
            algo_result = algo_result.cpu().numpy()[0, 0]
        if type(mask) == torch.Tensor:
            mask = mask.cpu().numpy()[0, 0]
        # Frobenius norm of the Hesse matrix
        diff = np.asarray(algo_result - gt, dtype='float64')
        dx = skf.scharr_v(diff)
        dy = skf.scharr_h(diff)
        dxx = skf.scharr_v(dx)
        dxy = skf.scharr_h(dx)
        dyy = skf.scharr_h(dy)
        dyx = skf.scharr_v(dy)
        bumpiness = np.sqrt(
            np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
        bumpiness = np.clip(bumpiness, 0, clip)
        # return bumpiness
        return np.mean(bumpiness[mask]) * factor

    for idx in tqdm(range(length)):
        k = name[idx, 0].split('/')[-1]

        with torch.no_grad():
            # get the gt and mask
            gt = torch.from_numpy(disp[idx][None, None].astype(np.float32)).to(
                args.device)
            mask = gt > -3.6
            mask_fp = (gt >= torch.min(focus_position)) & (
                gt <= torch.max(focus_position))

            input_dict['stack_rgb_img'] = torch.from_numpy(
                stack_rgb_img[idx].transpose(3, 0, 1, 2)[None].astype(
                    np.float32)).to(args.device)

            # Call network for inference
            start_time = time.time()
            infer_dict = net.module.inference(input_dict, args)
            losses['time'] += (time.time() - start_time) / length

            # calculate loss
            pred_disp = infer_dict['pred_disp']
            if pad_h:
                pred_disp = pred_disp[..., :-pad_h, :]
            if pad_w:
                pred_disp = pred_disp[..., :-pad_w]
            losses['MAE'] += L1loss_fn(pred_disp[mask], gt[mask]) / length
            losses['MSE'] += L2loss_fn(pred_disp[mask], gt[mask]) / length
            losses['RMSE'] += (L2loss_fn(pred_disp[mask], gt[mask]) **
                               0.5) / length
            # within focus position
            losses['MAE_fp'] += L1loss_fn(pred_disp[mask_fp],
                                          gt[mask_fp]) / length
            losses['MSE_fp'] += L2loss_fn(pred_disp[mask_fp],
                                          gt[mask_fp]) / length
            losses['RMSE_fp'] += (L2loss_fn(pred_disp[mask_fp], gt[mask_fp]) **
                                  0.5) / length
            losses['absRel_fp'] += torch.mean(
                abs(pred_disp[mask_fp] - gt[mask_fp]) /
                abs(gt[mask_fp])) / length
            losses['sqrRel_fp'] += torch.mean(
                ((pred_disp[mask_fp] - gt[mask_fp])**2) /
                abs(gt[mask_fp])) / length
            losses['Badpixel0.07'] += (
                (abs(pred_disp[mask_fp] - gt[mask_fp]) > 0.07).sum() /
                float(mask_fp.sum())) / length
            losses['Bumpiness'] += get_bumpiness(gt, pred_disp,
                                                 mask=mask_fp) / length
        # Dump results
        if args.outdir:
            imwrite(os.path.join(outpath, k + '.exr'),
                    pred_disp.cpu()[0, 0])
            cmap = cm.get_cmap('jet')
            color_img = cmap(
                ((torch.clamp(pred_disp, torch.min(focus_position),
                  torch.max(focus_position)) -
                  torch.min(focus_position)) /
                 (torch.max(focus_position) -
                  torch.min(focus_position))).cpu().numpy()[0, 0])[..., :3]
            imwrite(os.path.join(outpath, k + '.jpg'),
                    color_img,
                    quality=100)
            imwrite(os.path.join(outpath, k + '_rgb.jpg'),
                    infer_dict['pred_AiF_img'][0].cpu().numpy().transpose(
                        1, 2, 0),
                    quality=100)

    return losses


def Barron_2015_Blur_Dataset_testing(args):
    # Prepare all input rgb paths
    # load data
    args.stack_num = 15
    rgb_paths = [[] for i in range(args.stack_num)]
    disp_paths = []
    if args.txt is not None:
        with open(args.txt, 'r') as f:
            for line in tqdm(f.readlines(), desc='Load paths'):
                tmp = line.strip().split()
                for i in range(args.stack_num):
                    rgb_paths[i].append(tmp[i])
                disp_paths.append(tmp[-1])

    length = len(disp_paths)
    print('%d images in total.' % length)

    # Load trained checkpoint
    print('Loading checkpoint...', end='', flush=True)
    net, args_dict, args_model = utils.load_trained_model(args.pth)
    net = net.eval().to(args_dict['device'])
    for k, v in args_dict.items():
        if k not in args:
            setattr(args, k, v)
    print('done')

    # create output path
    if args.outdir:
        dirs = ['pred_disp', 'pred_AiF', 'pred_disp_jpg']
        outpath = os.path.join(args.outdir, args.dataset)
        for d in dirs:
            os.makedirs(os.path.join(outpath, d), exist_ok=True)

    # create losses dict
    losses = dict()
    losses['MAE'] = 0
    losses['MSE'] = 0
    losses['RMSE'] = 0
    losses['MAE_fp'] = 0
    losses['MSE_fp'] = 0
    losses['RMSE_fp'] = 0
    losses['logRMSE_fp'] = 0
    losses['absRel_fp'] = 0
    losses['sqrRel_fp'] = 0
    losses['time'] = 0
    # create loss function
    L1loss_fn = nn.L1Loss()
    L2loss_fn = nn.MSELoss()
    # focus position for two datasets
    if args.dataset == 'FLYINGTHINGS3D':
        focus_position = np.linspace(10, 100, 15)
    elif args.dataset == 'MIDDLEBURY':
        focus_position = np.linspace(10, 60, 15)
    else:
        raise NotImplementedError(
            "{} Dataset testing haven't implemented".format(args.dataset))

    # Loop through datasets
    for idx in tqdm(range(length)):
        k = os.path.split(disp_paths[idx])[0].split('/')[-1]

        with torch.no_grad():
            # get the gt and mask
            gt = torch.from_numpy(
                imread(disp_paths[idx])[None, None].astype(np.float32)).to(
                    args.device)
            mask = gt > 0
            mask_fp = (gt >= min(focus_position)) & (gt <= max(focus_position))

            input_dict = {
                'stack_rgb_img':
                torch.cat([
                    torch.from_numpy(
                        imread(x[idx]).transpose(2, 0, 1)[None].astype(
                            np.float32)).to(args.device).unsqueeze(2) / 255.
                    for x in rgb_paths
                ], 2),
                'focus_position':
                torch.from_numpy(focus_position).view(1, 1, args.stack_num).to(
                    args.device)
            }

            B, C, S, H, W = input_dict['stack_rgb_img'].shape
            pad_h = 32 - (H % 32) if H % 32 > 0 else 0
            pad_w = 32 - (W % 32) if W % 32 > 0 else 0
            input_dict['stack_rgb_img'] = nn.functional.pad(
                input_dict['stack_rgb_img'], (0, pad_w, 0, pad_h))

            # Call network for inference
            start_time = time.time()
            infer_dict = net.module.inference(input_dict, args)
            losses['time'] += (time.time() - start_time) / length

            # calculate loss
            pred_disp = infer_dict['pred_disp']
            pred_AiF = infer_dict['pred_AiF_img']

            if pad_h:
                pred_disp = pred_disp[..., :-pad_h, :]
                pred_AiF = pred_AiF[..., :-pad_h, :]
            if pad_w:
                pred_disp = pred_disp[..., :-pad_w]
                pred_AiF = pred_AiF[..., :-pad_w]

            # depth >0
            losses['MAE'] += L1loss_fn(pred_disp[mask], gt[mask]) / length
            losses['MSE'] += L2loss_fn(pred_disp[mask], gt[mask]) / length
            losses['RMSE'] += (L2loss_fn(pred_disp[mask], gt[mask]) **
                               0.5) / length
            # focus_min <= depth <= focus_max
            losses['MAE_fp'] += L1loss_fn(pred_disp[mask_fp],
                                          gt[mask_fp]) / length
            losses['MSE_fp'] += L2loss_fn(pred_disp[mask_fp],
                                          gt[mask_fp]) / length
            losses['RMSE_fp'] += (L2loss_fn(pred_disp[mask_fp], gt[mask_fp]) **
                                  0.5) / length
            losses['logRMSE_fp'] += (L2loss_fn(torch.log(
                pred_disp[mask_fp]), torch.log(gt[mask_fp]))**0.5) / length
            losses['absRel_fp'] += torch.mean(
                abs(pred_disp[mask_fp] - gt[mask_fp]) / gt[mask_fp])
            losses['sqrRel_fp'] += torch.mean(
                ((pred_disp[mask_fp] - gt[mask_fp])**2) / gt[mask_fp])
        # Dump results
        if args.outdir:
            imwrite(os.path.join(outpath, 'pred_disp', k + '.exr'),
                    pred_disp.cpu()[0, 0])
            # Jet and focus position normalization
            cmap = cm.get_cmap('jet')
            pred_disp = torch.clamp(pred_disp, min(focus_position), max(focus_position))
            pred_disp = (pred_disp - min(focus_position)) / (max(focus_position) - min(focus_position))
            color_img = cmap(
                pred_disp.cpu().numpy()[0, 0])[..., :3]
            imwrite(os.path.join(outpath, 'pred_disp_jpg', k + '.jpg'),
                    color_img,
                    quality=100)
            # AiF
            imwrite(os.path.join(outpath, 'pred_AiF', k + '.jpg'),
                    pred_AiF[0].cpu().numpy().transpose(
                        1, 2, 0),
                    quality=100)

    return losses


if __name__ == '__main__':
    # choose dataset
    if args.dataset == 'DEFOCUSNET':
        losses = DefocusNet_testing(args)
    elif args.dataset == 'DDFF':
        losses = DDFF_testing(args)
    elif args.dataset == 'HCI':
        losses = HCI_testing(args)
    elif args.dataset == 'MOBILE_DEPTH':
        losses = Mobile_Depth_testing(args)
    elif args.dataset == 'FLYINGTHINGS3D' or args.dataset == 'MIDDLEBURY':
        losses = Barron_2015_Blur_Dataset_testing(args)
    else:
        raise NotImplementedError(
            "{} Dataset testing haven't implemented".format(args.dataset))

    # eval with all depths
    for k, v in losses.items():
        if not k.endswith('_2'):
            print(k, str(v))
    # eval with depth <= 2 meters
    for k, v in losses.items():
        if k.endswith('_2'):
            print("ground truth <=2: ", k, str(v))
