import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, zeros_


class conv3d_bn(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.p = p
        self.conv3d = nn.Sequential(
            nn.Conv3d(self.in_ch,
                      self.out_ch,
                      kernel_size=self.k,
                      stride=self.s,
                      padding=self.p), nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv3d(x)


class trans3d_bn(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch=(64, 64),
                 k=(1, 1, 1),
                 s=(1, 1, 1),
                 p=(0, 0, 0)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.p = p
        self.trans3d = nn.Sequential(
            nn.ConvTranspose3d(self.in_ch,
                               self.out_ch[0],
                               kernel_size=self.k,
                               stride=self.s,
                               padding=self.p), nn.BatchNorm3d(self.out_ch[0]),
            nn.ReLU(inplace=True),
            conv3d_bn(self.out_ch[0],
                      self.out_ch[1],
                      k=(3, 3, 3),
                      s=(1, 1, 1),
                      p=(1, 1, 1)))

    def forward(self, x):
        return self.trans3d(x)


class Mixed(nn.Module):
    def __init__(self, in_ch=192, out_ch=(64, 96, 128, 16, 32, 32)):
        super().__init__()
        self.branch0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[0])

        self.branch1_0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[1])
        self.branch1_1 = conv3d_bn(in_ch=out_ch[1],
                                   out_ch=out_ch[2],
                                   k=(3, 3, 3),
                                   p=(1, 1, 1))

        self.branch2_0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[3])
        self.branch2_1 = conv3d_bn(in_ch=out_ch[3],
                                   out_ch=out_ch[4],
                                   k=(3, 3, 3),
                                   p=(1, 1, 1))

        self.branch3_0 = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1),
                                      padding=(1, 1, 1))
        self.branch3_1 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[5])

        self.output_channels = out_ch[0] + out_ch[2] + out_ch[
            4] + in_ch  # conv1, conv2, conv3, max1

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1_1(self.branch1_0(x))
        b2 = self.branch2_1(self.branch2_0(x))
        b3 = self.branch3_1(self.branch3_0(x))

        return torch.cat([b0, b1, b2, b3], 1)


class AiFDepthNet(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=1,
                 n_stack=10,
                 disp_w=1,
                 aif_w=0,
                 smooth_w=0,
                 focal_length=521.4052,
                 baseline=0.00027087,
                 depth_min=0.1,
                 depth_max=3,
                 mask_range=False,
                 normalize_attention=False,
                 disp_depth='depth',
                 stage2='attention'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_stack = n_stack
        # loss weight
        self.disp_w = disp_w
        self.AiF_w = aif_w
        self.SMOOTH_W = smooth_w
        self.STAGE2 = stage2.upper()
        # camera configuration
        self.f_l = focal_length  # focal length
        self.b = baseline  # baseline
        self.DEPTH_MIN = depth_min
        self.DEPTH_MAX = depth_max
        self.depth_range = [self.DEPTH_MIN, self.DEPTH_MAX]
        self.DISP_DEPTH = disp_depth
        if self.DISP_DEPTH == 'disp':
            self.disparity_range = [
                self.b * self.f_l / x for x in self.depth_range
            ]
            self.d_layers = np.linspace(self.disparity_range[0],
                                        self.disparity_range[1], self.n_stack)
        else:
            self.d_layers = np.linspace(self.depth_range[0],
                                        self.depth_range[1], self.n_stack)
        self.MASK_RANGE = mask_range
        self.NORMALIZE_ATTENTION = normalize_attention
        """
        Conv3d_1a_7x7
        """
        self.conv3d_1a = conv3d_bn(n_channels,
                                   out_ch=64,
                                   k=(7, 7, 7),
                                   s=(1, 2, 2),
                                   p=(3, 3, 3))
        """
        MaxPool3d_2a_3X3
        """
        self.max3d_2a = nn.MaxPool3d((1, 3, 3),
                                     stride=(1, 2, 2),
                                     padding=(0, 1, 1))
        """
        Conv3d_2b_1x1
        """
        self.conv3d_2b = conv3d_bn(in_ch=64, out_ch=64)
        """
        Conv3d_2c_3x3
        """
        self.conv3d_2c = conv3d_bn(in_ch=64,
                                   out_ch=192,
                                   k=(3, 3, 3),
                                   p=(1, 1, 1))
        """
        MaxPool3d_3a_3X3
        """
        self.max3d_3a = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                     stride=(1, 2, 2),
                                     padding=(0, 1, 1))
        """
        Mixed_3b
        """
        self.Mixed_3b = Mixed(in_ch=192,
                              out_ch=(64, 96, 128, 16, 32,
                                      32))  # out_ch = 64+128+32+192=256
        """
        Mixed_3c
        """
        self.Mixed_3c = Mixed(in_ch=256, out_ch=(128, 128, 192, 32, 96,
                                                 64))  # out_ch = 480
        """
        Mixed 4a, 4b ,4c, 4d, 4e, 4f
        """
        self.max3d_4a = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                     stride=(1, 2, 2),
                                     padding=(0, 1, 1))  # 1/8 size
        self.Mixed_4b = Mixed(in_ch=480, out_ch=(192, 96, 208, 16, 48,
                                                 64))  # out_ch = 512
        self.Mixed_4c = Mixed(in_ch=512, out_ch=(160, 112, 224, 24, 64,
                                                 64))  # out_ch = 512
        self.Mixed_4d = Mixed(in_ch=512, out_ch=(128, 128, 256, 24, 64,
                                                 64))  # out_ch = 512
        self.Mixed_4e = Mixed(in_ch=512, out_ch=(112, 144, 288, 32, 64,
                                                 64))  # out_ch = 528
        self.Mixed_4f = Mixed(in_ch=528, out_ch=(256, 160, 320, 32, 128,
                                                 128))  # out_ch = 832
        """
        MaxPool3d_5a_2x2
        """
        self.max3d_5a = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        """
        Mixed 5b ,5c
        """
        self.Mixed_5b = Mixed(in_ch=832, out_ch=(256, 160, 320, 32, 128,
                                                 128))  # out_ch = 832
        self.Mixed_5c = Mixed(in_ch=832, out_ch=(384, 192, 384, 48, 128,
                                                 128))  # out_ch = 1024

        # ---------
        # Decoder
        # ---------
        """
        Upconv1
        """
        self.up_5c = trans3d_bn(in_ch=1024,
                                out_ch=(64, 64),
                                k=(3, 4, 4),
                                s=(1, 2, 2),
                                p=(1, 1, 1))  # 64, 16, 14, 14
        self.up_4f = conv3d_bn(in_ch=832, out_ch=64)
        """
        Upconv2
        """
        self.up_5c4f = trans3d_bn(in_ch=128,
                                  out_ch=(64, 64),
                                  k=(3, 4, 4),
                                  s=(1, 2, 2),
                                  p=(1, 1, 1))  # 64, 32, 28, 28
        self.up_3c = conv3d_bn(in_ch=480, out_ch=64)
        """
        Upconv3
        """
        self.up_5c4f3c = trans3d_bn(
            in_ch=128, out_ch=(32, 32), k=(3, 4, 4), s=(1, 2, 2),
            p=(1, 1, 1))  # 32, 32, 56, 56, default kernel_size = 4
        self.up_2c = conv3d_bn(in_ch=192, out_ch=32)
        """
        Upconv4
        """
        self.up_5c4f3c2c = trans3d_bn(
            in_ch=64, out_ch=(32, 16), k=(3, 4, 4), s=(1, 2, 2),
            p=(1, 1, 1))  # 32, 32, 112, 112, default kernel_size = 4
        self.up_1a = conv3d_bn(in_ch=64, out_ch=16)
        """
        Upconv5
        """
        self.final_up = nn.ConvTranspose3d(32,
                                           32,
                                           kernel_size=(3, 4, 4),
                                           stride=(1, 2, 2),
                                           padding=(1, 1,
                                                    1))  # 32, 64, 224, 224
        self.out = nn.Conv3d(32,
                             self.n_classes,
                             kernel_size=(3, 3, 3),
                             stride=(1, 1, 1),
                             padding=(1, 1, 1))  # n*3, 64, 224, 224
        """
        Direct output
        """
        if self.STAGE2 == 'DIRECT':
            self.d_out = nn.Conv2d(self.n_stack,
                                   1,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=(0, 0))
            self.AiF_out = nn.Sequential(
                nn.Conv2d(self.n_stack,
                          3,
                          kernel_size=(1, 1),
                          stride=(1, 1),
                          padding=(0, 0)), nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, input_dict, args):
        # init
        assert args.stack_num == self.n_stack, print(
            "stack num: {}, n_stack: {}".format(str(args.stack_num),
                                                str(self.n_stack)))
        stack_rgb = input_dict['stack_rgb_img'].to(args.device)  # focal stack
        # focus position
        if 'focus_position' in input_dict:
            self.d_layers = input_dict['focus_position']

        if self.n_channels == 4:
            B, C, S, H, W = stack_rgb.shape
            stack_rgb = torch.cat([
                stack_rgb, (torch.arange(1, S + 1) / float(S)).view(
                    1, 1, S, 1, 1).repeat(B, 1, 1, H, W).to(args.device)
            ], 1)

        # run model
        outputs = self.fit(stack_rgb, args)
        losses, outputs = self.compute_loss(outputs, input_dict, args)

        return losses, outputs

    def fit(self, x, args):
        """
        init
        """
        outputs = dict()
        # --------
        # Encoder
        # --------
        """
        Conv 1a
        """
        conv1a = self.conv3d_1a(x)
        """
        Max3d 2a
        Conv 2b, 2c
        """
        conv2c = self.conv3d_2c(self.conv3d_2b(self.max3d_2a(conv1a)))
        """
        Max3d 3a
        Mixed 3b, 3c
        """
        Mix3c = self.Mixed_3c(self.Mixed_3b(self.max3d_3a(conv2c)))
        """
        Max3d 4a
        Mixed 4b, 4c, 4d, 4e, 4f
        """
        Mix4f = self.Mixed_4f(
            self.Mixed_4e(
                self.Mixed_4d(
                    self.Mixed_4c(self.Mixed_4b(self.max3d_4a(Mix3c))))))
        """
        Max3d 5a,
        Mixed 5b, bc
        """
        Mix5c = self.Mixed_5c(self.Mixed_5b(self.max3d_5a(Mix4f)))

        # ---------
        # Decoder
        # ---------
        """
        Upconv1
        """
        Up_5c = self.up_5c(Mix5c)
        Up_4f = self.up_4f(Mix4f)
        Cat_5c4f = torch.cat([Up_5c, Up_4f], 1)
        """
        Upconv2
        """
        Up_5c4f = self.up_5c4f(Cat_5c4f)
        Up_3c = self.up_3c(Mix3c)
        Cat_5c4f3c = torch.cat([Up_5c4f, Up_3c], 1)
        """
        Upconv3
        """
        Up_5c4f3c = self.up_5c4f3c(Cat_5c4f3c)
        Up_2c = self.up_2c(conv2c)
        Cat_5c4f3c2c = torch.cat([Up_5c4f3c, Up_2c], 1)
        """
        Upconv4
        """
        Up_5c4f3c2c = self.up_5c4f3c2c(Cat_5c4f3c2c)
        Up_1a = self.up_1a(conv1a)
        Cat_final = torch.cat([Up_5c4f3c2c, Up_1a], 1)
        """
        Upconv5
        """
        Up_final = self.final_up(Cat_final)
        out = self.out(Up_final)
        """
        result
        """
        B, C, S, H, W = out.shape

        if self.STAGE2 == 'ATTENTION':
            # Attention construction
            if self.n_classes == 2:
                # disp/depth attention
                if self.NORMALIZE_ATTENTION:
                    d_attention = F.softplus(out[:, 0, ...])
                    d_attention = d_attention / d_attention.sum(axis=-3,
                                                                keepdim=True)
                else:
                    d_attention = F.softmax(out[:, 0, ...], -3)
                # disparity output
                if type(self.d_layers) == list or type(
                        self.d_layers) == np.ndarray:
                    d_maps = torch.from_numpy(
                        np.array(self.d_layers).reshape(1, S, 1, 1).repeat(
                            (B, 1, H, W))).float().to(args.device)
                else:
                    d_maps = self.d_layers.view(B, S, 1, 1).repeat(
                        (1, 1, H, W)).float().to(args.device)

                d_out = torch.sum(d_attention * d_maps, dim=-3,
                                  keepdim=False).view(B, 1, H, W)

                # AiF attention
                if self.NORMALIZE_ATTENTION:
                    AiF_attention = F.softplus(out[:, 1, ...])
                    AiF_attention = AiF_attention / AiF_attention.sum(
                        axis=-3, keepdim=True)
                else:
                    AiF_attention = F.softmax(out[:, 1, ...], -3)
                # AiF output
                AiF = AiF_attention.view(B, 1, S, H, W).repeat(
                    (1, 3, 1, 1, 1)) * x[:, :3]
                AiF = torch.sum(AiF, dim=-3, keepdim=False)
            elif self.n_classes == 1:
                # share attention
                if self.NORMALIZE_ATTENTION:
                    d_attention = F.softplus(out)
                    d_attention = d_attention / d_attention.sum(axis=-3,
                                                                keepdim=True)
                    AiF_attention = F.softmax(out, -3)
                else:
                    d_attention = F.softmax(out, -3)
                    AiF_attention = F.softmax(out, -3)
                # disparity output
                if type(self.d_layers) == list:
                    d_maps = torch.from_numpy(
                        np.array(self.d_layers).reshape(1, 1, S, 1, 1)).repeat(
                            (B, 1, 1, H, W)).float().to(args.device)
                else:
                    d_maps = self.d_layers.view(B, 1, S, 1, 1).repeat(
                        (1, 1, 1, H, W)).float().to(args.device)

                d_out = torch.sum(d_attention * d_maps, dim=-3, keepdim=False)
                # AiF output
                AiF = AiF_attention.repeat((1, 3, 1, 1, 1)) * x[:, :3]
                AiF = torch.sum(AiF, dim=-3, keepdim=False)
        elif self.STAGE2 == 'DIRECT':
            if self.n_classes == 2:
                d_out = self.d_out(out[:, 0])
                AiF = self.AiF_out(out[:, 1])
            elif self.n_classes == 1:
                d_out = self.d_out(out[:, 0])
                AiF = self.AiF_out(out[:, 0])

        outputs['pred_{}'.format(self.DISP_DEPTH)] = d_out
        outputs['pred_AiF_img'] = AiF

        return outputs

    def compute_loss(self, outputs, input_dict, args):
        # init
        losses = dict()
        d_out, AiF = outputs['pred_{}'.format(
            self.DISP_DEPTH)], outputs['pred_AiF_img']

        if args.task == 'D_FS':
            # init
            gt_d = input_dict[self.DISP_DEPTH]

            # mask gt disp/depth
            if self.MASK_RANGE:
                mask = (gt_d >= torch.min(self.d_layers)) & (gt_d <= torch.max(
                    self.d_layers))
            else:
                mask = gt_d > 0
            mask.detach_()

            gt_h, gt_w = gt_d.shape[2:]
            pred_h, pred_w = d_out.shape[2:]
            if pred_h > gt_h:
                crop_h = pred_h - gt_h
                d_out = d_out[..., :-crop_h, :]
                AiF = AiF[..., :-crop_h, :]
            if pred_w > gt_w:
                crop_w = pred_w - gt_w
                d_out = d_out[..., :-crop_w]
                AiF = AiF[..., :-crop_w]

            # Loss Compute
            losses[self.DISP_DEPTH] = F.l1_loss(d_out[mask],
                                                gt_d[mask],
                                                reduction='mean')
            # show MSE
            with torch.no_grad():
                losses['disp_MSE'] = F.mse_loss(d_out[mask],
                                                gt_d[mask],
                                                reduction='mean')
            losses['total'] = self.disp_w * losses[self.DISP_DEPTH]

        elif args.task == 'A_FS':
            # init
            gt_AiF = input_dict['AiF_img'].to(args.device)

            # crop prediction
            gt_h, gt_w = gt_AiF.shape[2:]
            pred_h, pred_w = d_out.shape[2:]
            if pred_h > gt_h:
                crop_h = pred_h - gt_h
                d_out = d_out[..., :-crop_h, :]
                AiF = AiF[..., :-crop_h, :]
            if pred_w > gt_w:
                crop_w = pred_w - gt_w
                d_out = d_out[..., :-crop_w]
                AiF = AiF[..., :-crop_w]

            # compute loss
            losses['AiF'] = F.l1_loss(AiF, gt_AiF, reduction='mean')
            # # smoothness
            abs_fn = lambda x: x**2
            edge_constant = 150.
            img_gx, img_gy = self.image_grads(gt_AiF)
            weights_x = torch.exp(-torch.mean(
                abs_fn(edge_constant * img_gx), axis=1, keepdims=True))
            weights_y = torch.exp(-torch.mean(
                abs_fn(edge_constant * img_gy), axis=1, keepdims=True))

            # Compute second derivatives of the predicted smoothness.
            d_gx, d_gy = self.image_grads(d_out)

            # Compute weighted smoothness
            losses['smooth'] = (
                torch.mean(weights_x * self.robust_l1(d_gx)) +
                torch.mean(weights_y * self.robust_l1(d_gy))) / 2.
            # end smoothness
            losses['total'] = self.AiF_w * losses[
                'AiF'] + self.SMOOTH_W * losses['smooth']

        elif args.task == 'DA_FS':
            # init
            gt_d = input_dict[self.DISP_DEPTH]
            gt_AiF = input_dict['AiF_img'].to(args.device)

            # mask gt disp/depth
            if self.MASK_RANGE:
                mask = (gt_d >= torch.min(self.d_layers)) & (gt_d <= torch.max(
                    self.d_layers))
            else:
                mask = gt_d > 0
            mask.detach_()

            # crop prediction
            gt_h, gt_w = gt_d.shape[2:]
            pred_h, pred_w = d_out.shape[2:]
            if pred_h > gt_h:
                crop_h = pred_h - gt_h
                d_out = d_out[..., :-crop_h, :]
                AiF = AiF[..., :-crop_h, :]
                gt_AiF = gt_AiF[..., :-crop_h, :]
            if pred_w > gt_w:
                crop_w = pred_w - gt_w
                d_out = d_out[..., :-crop_w]
                AiF = AiF[..., :-crop_w]
                gt_AiF = gt_AiF[..., :-crop_w]

            # Loss Compute
            losses[self.DISP_DEPTH] = F.l1_loss(d_out[mask],
                                                gt_d[mask],
                                                reduction='mean')
            losses['AiF'] = F.l1_loss(AiF, gt_AiF, reduction='mean')
            # # smoothness
            abs_fn = lambda x: x**2
            edge_constant = 150.
            img_gx, img_gy = self.image_grads(gt_AiF)
            weights_x = torch.exp(-torch.mean(
                abs_fn(edge_constant * img_gx), axis=1, keepdims=True))
            weights_y = torch.exp(-torch.mean(
                abs_fn(edge_constant * img_gy), axis=1, keepdims=True))

            # Compute second derivatives of the predicted smoothness.
            d_gx, d_gy = self.image_grads(d_out)

            # Compute weighted smoothness
            losses['smooth'] = (
                torch.mean(weights_x * self.robust_l1(d_gx)) +
                torch.mean(weights_y * self.robust_l1(d_gy))) / 2.
            # end smoothness

            losses[
                'total'] = self.AiF_w * losses['AiF'] + self.disp_w * losses[
                    self.DISP_DEPTH] + self.SMOOTH_W * losses['smooth']

        else:
            raise NotImplementedError()

        return losses, outputs

    def inference(self, input_dict, args):
        # init
        stack_rgb = input_dict['stack_rgb_img'].to(args.device)  # focal stack
        # focus position
        if 'focus_position' in input_dict:
            self.d_layers = input_dict['focus_position']

        if self.n_channels == 4:
            B, C, S, H, W = stack_rgb.shape
            stack_rgb = torch.cat([
                stack_rgb, (torch.arange(1, S + 1) / float(S)).view(
                    1, 1, S, 1, 1).repeat(B, 1, 1, H, W).to(args.device)
            ], 1)

        # run model
        outputs = self.fit(stack_rgb, args)

        return outputs

    def image_grads(self, image_batch, stride=1):
        image_batch_gh = image_batch[..., stride:, :] - image_batch[
            ..., :-stride, :]
        image_batch_gw = image_batch[..., stride:] - image_batch[..., :-stride]
        return image_batch_gh, image_batch_gw

    def robust_l1(self, x):
        """Robust L1 metric."""
        return (x**2 + 0.001**2)**0.5
