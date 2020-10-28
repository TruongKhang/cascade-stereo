import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import time
import sys
sys.path.append("..")
# from utils import local_pcd
from models.utils.disp2prob import LaplaceDisp2Prob


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=False, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1







# def resample_vol_cuda(src_vol, src_proj, ref_proj, cam_intrinsic=None,
#                       d_candi=None, d_candi_new=None,
#                       padding_value=0., output_tensor=False,
#                       is_debug=False, PointsDs_ref_cam_coord_in=None):
#     r'''
#
#     if d_candi_new is not None:
#     d_candi : candidate depth values for the src view;
#     d_candi_new : candidate depth values for the ref view. Usually d_candi_new is different from d_candi
#     '''
#     assert d_candi is not None, 'd_candi should be some np.array object'
#
#     N, D, H, W = src_vol.shape
#     N = 1
#     hhfov, hvfov = \
#             math.radians(cam_intrinsic['hfov']) * .5, math.radians(cam_intrinsic['vfov']) * .5
#
#     # --- 0. Get the sampled points in the ref. view --- #
#     if PointsDs_ref_cam_coord_in is None:
#         PointsDs_ref_cam_coord = torch.zeros(N, D, H, W, 3)
#         if d_candi_new is not None:
#             d_candi_ = d_candi_new
#         else:
#             d_candi_ = d_candi
#
#         for idx_d, d in enumerate(d_candi_):
#             PointsDs_ref_cam_coord[0, idx_d, :, :, :] \
#                     = d * torch.FloatTensor(cam_intrinsic['unit_ray_array'])
#         PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.cuda(0)
#     else:
#         PointsDs_ref_cam_coord = PointsDs_ref_cam_coord_in
#
#     if d_candi_new is not None:
#         z_max, z_min = d_candi.max(), d_candi.min()
#     else:
#         z_max = torch.max(PointsDs_ref_cam_coord[0, :, :, :, 2])
#         z_min = torch.min(PointsDs_ref_cam_coord[0, :, :, :, 2])
#
#     z_half = (z_max + z_min) * .5
#     z_radius = (z_max - z_min) * .5
#
#     # --- 1. Coordinate transform --- #
#     PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.reshape((-1, 3)).transpose(0,1)
#     PointsDs_ref_cam_coord = torch.cat((PointsDs_ref_cam_coord, torch.ones(1, PointsDs_ref_cam_coord.shape[1]).cuda(0)), dim=0)
#
#     src_cam_extM = rel_extM
#     PointsDs_src_cam_coord = src_cam_extM.matmul(PointsDs_ref_cam_coord)
#
#     # transform into range [-1, 1] for all dimensions #
#     PointsDs_src_cam_coord[0, :] = PointsDs_src_cam_coord[0,:] / (PointsDs_src_cam_coord[2,:] +1e-10) / math.tan( hhfov)
#     PointsDs_src_cam_coord[1, :] = PointsDs_src_cam_coord[1,:] / (PointsDs_src_cam_coord[2,:] +1e-10) / math.tan( hvfov)
#     PointsDs_src_cam_coord[2, :] = (PointsDs_src_cam_coord[2,:] -  z_half ) / z_radius
#
#     # reshape to N x OD x OH x OW x 3 #
#     PointsDs_src_cam_coord = PointsDs_src_cam_coord / (PointsDs_src_cam_coord[3,:].unsqueeze_(0) + 1e-10 )
#     PointsDs_src_cam_coord = PointsDs_src_cam_coord[:3, :].transpose(0,1).reshape((N, D, H, W, 3))
#
#     # --- 2. Re-sample --- #
#     src_vol_th = src_vol.unsqueeze(1)
#     src_vol_th_ = _set_vol_border(src_vol_th, padding_value)
#     res_vol_th = torch.squeeze(torch.squeeze(
#         F.grid_sample(src_vol_th_, PointsDs_src_cam_coord, mode='bilinear', padding_mode='border'), dim=0), dim=0)
#
#     if is_debug:
#         return res_vol_th, PointsDs_src_cam_coord, src_vol_th
#     else:
#         return res_vol_th


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        if self.last_layer:
            self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, padding=True):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        self.dc_feature = nn.Sequential(nn.Conv2d(self.input_channels[0], 16, kernel_size=3, padding=1),
                                        nn.ReLU())
        self.img_feature = nn.Sequential(nn.Conv2d(self.input_channels[1], 16, kernel_size=3, padding=1),
                                         nn.ReLU())

        layers = []
        output_dim = 32
        for i in range(len(self.num_filters[:-1])):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = 32 if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))
        self.last_downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.latent_mu = nn.Sequential(nn.Conv2d(output_dim, self.num_filters[-1]//2, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.num_filters[-1]//2, self.num_filters[-1]//2, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.latent_sigma = nn.Sequential(nn.Conv2d(output_dim, self.num_filters[-1] // 2, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(self.num_filters[-1] // 2, self.num_filters[-1] // 2, kernel_size=3,
                                                    padding=1),
                                          nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)
        self.dc_feature.apply(init_weights)
        self.img_feature.apply(init_weights)
        self.latent_mu.apply(init_weights)
        self.latent_sigma.apply(init_weights)

    def forward(self, inputs):
        # print(input.size(), self.input_channels)
        dc, img_feat = inputs
        dc = self.dc_feature(dc)
        img_feat = self.img_feature(img_feat)
        output = self.layers(torch.cat((dc, img_feat), dim=1))
        output = self.last_downsample(output)
        mu = self.latent_mu(output)
        sigma = self.latent_sigma(output)
        return mu, sigma


class Decoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, laten_dim, padding=True):
        super(Decoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        layers = [nn.Conv2d(laten_dim, input_channels, kernel_size=3, padding=1)]
        for i in range(len(self.num_filters)-1):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            layers.append(nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU(inplace=True))

            if i < (len(self.num_filters)-2):
                for _ in range(no_convs_per_block - 1):
                    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                    layers.append(nn.ReLU(inplace=True))

        self.final_layer = nn.Sequential(nn.Conv2d(output_dim, self.num_filters[-1], kernel_size=3, padding=1),
                                         nn.Softmax(dim=1))
        # self.variance = nn.Conv2d(output_dim, self.num_filters[-1], 1)

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)
        self.final_layer.apply(init_weights)
        # self.variance.apply(init_weights)

    def forward(self, inputs, depth_values=None, variance=None):
        output = self.layers(inputs)
        #depth = self.final_layer(output)
        #var = self.variance(output) * variance
        prob_volume = self.final_layer(output) #LaplaceDisp2Prob(depth_values, depth, variance).getProb()
        return prob_volume.unsqueeze(1) # output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block)
        # self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, kernel_size=3, padding=1)
        self.fc_mu = nn.Conv2d(num_filters[-1]//2, self.latent_dim, kernel_size=1)
        self.fc_sigma = nn.Sequential(nn.Conv2d(num_filters[-1]//2, self.latent_dim, kernel_size=1),
                                      nn.Softplus(beta=0.01))

        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.fc_mu.bias)
        nn.init.kaiming_normal_(self.fc_sigma.weight)
        nn.init.normal_(self.fc_sigma.bias)

    def forward(self, inputs): #, conf=None):

        latent_mu, latent_sigma = self.encoder(inputs)

        # We only want the mean of the resulting hxw image
        latent_mu = torch.mean(latent_mu, dim=2, keepdim=True)
        latent_mu = torch.mean(latent_mu, dim=3, keepdim=True)
        latent_sigma = torch.mean(latent_sigma, dim=2, keepdim=True)
        latent_sigma = torch.mean(latent_sigma, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu = self.fc_mu(latent_mu)
        sigma = self.fc_sigma(latent_sigma)

        mu = mu.squeeze(3).squeeze(2)
        sigma = sigma.squeeze(3).squeeze(2)
        # mu = mu.permute(0, 2, 3, 1)
        # log_sigma = log_sigma.permute(0, 2, 3, 1)

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=sigma), 1)
        return dist


class GenerationNet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=5, output_channels=8, num_filters=[16, 32, 64, 96, 128], beta=10.0,
                 latent_dim=64):
        super(GenerationNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.no_convs_per_block = 2
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.latent_dim = latent_dim
        self.z_prior_sample = 0

        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                             self.latent_dim) #.to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                 self.latent_dim, posterior=True) #.to(device)
        num_channels_decoder = num_filters[:-1][::-1] + [self.output_channels]
        self.generator = Decoder(self.num_filters[-1], num_channels_decoder, 2, self.latent_dim)

        self.posterior_latent_space = None
        self.prior_latent_space = None

    def forward(self, img_feature, costvol, gt_costvol=None, training=True): #img, prior_depth, conf, gt_depth=None, gt_conf=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(torch.cat((img_feature, gt_costvol), dim=1)) #torch.cat((img, gt_depth, gt_conf), dim=1))
        self.prior_latent_space = self.prior.forward(torch.cat((img_feature, costvol), dim=1)) #torch.cat((img, prior_depth, conf), dim=1))
        # self.unet_features = self.unet.forward(patch, False)

    def sample(self, testing=False, depth_values=None, variance=None, num_samples=1):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
            cost_volume = self.generator.forward(z_prior.permute(0, 3, 1, 2), depth_values, variance)
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            cost_volume = []
            for _ in range(num_samples):
                z_prior = self.prior_latent_space.sample()
                # self.z_prior_sample = z_prior
                cost_volume.append(self.generator.forward(z_prior.permute(0, 3, 1, 2), depth_values, variance))
            cost_volume = torch.cat(cost_volume, dim=1)
        return cost_volume #self.generator.forward(z_prior.permute(0, 3, 1, 2), depth_values, variance)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=True, z_posterior=None, depth_values=None, variance=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        # print(z_posterior.size())
        return self.generator.forward(z_posterior.permute(0, 3, 1, 2), depth_values, variance)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, analytic_kl=True, reconstruct_posterior_mean=False, depth_values=None, variance=None):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        # criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()

        kl = self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
        #kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        reconst_vol = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False,
                                       z_posterior=z_posterior, depth_values=depth_values, variance=variance)

        # reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        # self.reconstruction_loss = torch.sum(reconstruction_loss)
        # self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return reconst_vol, kl #-(self.reconstruction_loss + self.beta * self.kl)


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples



if __name__ == "__main__":
    # some testing code, just IGNORE it
    # import sys
    # sys.path.append("../")
    # from datasets import find_dataset_def
    # from torch.utils.data import DataLoader
    # import numpy as np
    # import cv2
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt
    #
    # # MVSDataset = find_dataset_def("colmap")
    # # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    # #                      128, interval_scale=1.06, max_h=1250, max_w=1024)
    #
    # MVSDataset = find_dataset_def("dtu_yao")
    # num_depth = 48
    # dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
    #                      3, num_depth, interval_scale=1.06 * 192 / num_depth)
    #
    # dataloader = DataLoader(dataset, batch_size=1)
    # item = next(iter(dataloader))
    #
    # imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # # imgs = item["imgs"][:, :, :, :, :]
    # proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    # depth_values = item["depth_values"]     #(B, D)
    #
    # imgs = torch.unbind(imgs, 1)
    # proj_matrices = torch.unbind(proj_matrices, 1)
    # ref_img, src_imgs = imgs[0], imgs[1:]
    # ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view
    #
    # src_proj_new = src_proj[:, 0].clone()
    # src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    # ref_proj_new = ref_proj[:, 0].clone()
    # ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])



    # warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)
    #
    # ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    # cv2.imwrite('../tmp/ref.png', ref_img_np)
    # cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    #
    # for i in range(warped_imgs.shape[2]):
    #     warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
    #     img_np = warped_img[0].detach().cpu().numpy()
    #     img_np = img_np[:, :, ::-1] * 255
    #
    #     alpha = 0.5
    #     beta = 1 - alpha
    #     gamma = 0
    #     img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
    #     cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))

    generator = GenerationNet(5, 8)
    generator.cuda()
    img = torch.rand(2, 3, 480, 640).cuda()
    prior_depth = torch.rand(2, 1, 480, 640).cuda()
    conf = (prior_depth > 0.5).float().cuda()
    gt_depth = torch.rand(2, 1, 480, 640).cuda()
    gt_conf = (gt_depth > 0.5).float().cuda()
    generator(img, prior_depth, conf, gt_depth, gt_conf)
    costvol, kl = generator.elbo() #generator.reconstruct()

    print(costvol.size(), kl)
