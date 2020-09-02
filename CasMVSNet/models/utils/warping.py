import torch
import torch.nn.functional as F

import MYTH


def homo_warping_3D(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping_2D(src_depths, src_cfds, src_proj, ref_proj):
    src_p = src_proj[:, :3, :4]
    ref_p = ref_proj[:, :3, :4]
    camera_params = torch.cat((ref_p.unsqueeze(1), src_p.unsqueeze(1)), 1)
    depths = src_depths.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    cfds = src_cfds.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    warped_depths, warped_cfds, _ = MYTH.DepthColorAngleReprojectionNeighbours.apply(depths, cfds, camera_params, 1.0)
    warped_depths = warped_depths[:, -1]
    warped_cfds = warped_cfds[:, -1]
    return warped_depths, warped_cfds


def resample_vol(src_vol, src_proj, ref_proj, depth_values, prev_depth_values=None, begin_video=None):
    # src_vol: [B, Ndepth, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, Ndepth, H, W]
    batch = src_vol.shape[0]
    num_depth = depth_values.shape[1]
    height, width = src_vol.shape[2], src_vol.shape[3]

    if prev_depth_values is None:
        prev_depth_values = depth_values
    elif begin_video is not None:
        prev_depth_values[begin_video] = depth_values[begin_video]
    depth_min = prev_depth_values[:, 0] # [B, H, W]
    depth_max = prev_depth_values[:, -1] # [B, H, W]
    depth_half = (depth_max + depth_min) * .5
    depth_radius = (depth_max - depth_min) * .5
    depth_half, depth_radius = depth_half.view(batch, 1, -1), depth_radius.view(batch, 1, -1)

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        # print(proj.size())
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_vol.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_vol.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
        # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_z_normalized = (proj_xyz[:, 2, :, :] - depth_half) / depth_radius  # [B, Ndepth, H*W]
        proj_xyz_normalized = torch.stack((proj_x_normalized, proj_y_normalized, proj_z_normalized), dim=3)  # [B, Ndepth, H*W, 3]
        grid = proj_xyz_normalized

    src_vol_new = set_vol_border(src_vol.unsqueeze(1), 0) #math.log(1.0/num_depth))
    warped_src_vol = F.grid_sample(src_vol_new, grid.view(batch, num_depth, height, width, 3),
                                   mode='bilinear',
                                   padding_mode='border')
    warped_src_vol = warped_src_vol.squeeze(1).view(batch, num_depth, height, width)
    # warped_src_vol = F.normalize(warped_src_vol, p=1, dim=1) #F.log_softmax(warped_src_vol, dim=1)
    # print(warped_src_vol.min(), warped_src_vol.max())

    return warped_src_vol #warped_src_vol.clamp(min=-1000, max=0)


def set_vol_border(vol, border_val):
    '''
    inputs:
    vol - a torch tensor in 3D: N x C x D x H x W
    border_val - a float, the border value
    '''
    vol_ = vol + 0.
    vol_[:, :, 0, :, :] = border_val
    vol_[:, :, :, 0, :] = border_val
    vol_[:, :, :, :, 0] = border_val
    vol_[:, :, -1, :, :] = border_val
    vol_[:, :, :, -1, :] = border_val
    vol_[:, :, :, :, -1] = border_val

    return vol_