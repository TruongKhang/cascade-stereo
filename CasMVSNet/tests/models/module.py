import unittest
import numpy as np
import torch
import torch.functional as F

from models.module import *


class TestWarpingVolume(unittest.TestCase):
    def test_shape(self):
        height, width = 480, 640
        batch_size = 2
        src_pose = np.array([[0.970263, 0.00747983, 0.241939, -191.02],
                            [-0.0147429, 0.999493, 0.0282234, 3.28832],
                            [-0.241605, -0.030951, 0.969881, 22.5401],
                            [0.0, 0.0, 0.0, 1.0]])
        ref_pose = np.array([[0.802256, -0.439347, 0.404178, -291.419],
                             [0.427993, 0.895282, 0.123659, -77.0495],
                             [-0.416183, 0.073779, 0.906283, 71.2762],
                             [0.0, 0.0, 0.0, 1.0]])
        intrinsic_matrix = np.array([[2892.33, 0, 320], [0, 2883.18, 240], [0, 0, 1]])

        src_matrix, ref_matrix = np.eye(4), np.eye(4)

        src_matrix[:3, :4] = np.dot(intrinsic_matrix, src_pose[:3])
        ref_matrix[:3, :4] = np.dot(intrinsic_matrix, ref_pose[:3])
        src_matrix = torch.tensor(src_matrix).unsqueeze(0).repeat(batch_size, 1, 1)
        ref_matrix = torch.tensor(ref_matrix).unsqueeze(0).repeat(batch_size, 1, 1)

        depth_min, depth_interval, num_depth = 425, 2.5, 64
        depth_max = depth_min + num_depth * depth_interval
        depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
        depth_values = torch.tensor(depth_values).unsqueeze(0).repeat(batch_size, 1)
        depth_values = depth_values.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)

        src_vol = torch.rand(batch_size, num_depth, height, width)
        src_vol = F.softmax(src_vol, dim=1)

        warped_vol = resample_vol(src_vol.float().cuda(), src_matrix.float().cuda(), ref_matrix.float().cuda(),
                                  depth_values.float().cuda())

        self.assertEqual(warped_vol.size(), (batch_size, num_depth, height, width), 'incorrect warping size')

    def test_volume1(self):
        height, width = 512, 640
        batch_size = 2
        src_pose = np.array([[0.970263, 0.00747983, 0.241939, -191.02],
                             [-0.0147429, 0.999493, 0.0282234, 3.28832],
                             [-0.241605, -0.030951, 0.969881, 22.5401],
                             [0.0, 0.0, 0.0, 1.0]])
        ref_pose = np.array([[0.802256, -0.439347, 0.404178, -291.419],
                             [0.427993, 0.895282, 0.123659, -77.0495],
                             [-0.416183, 0.073779, 0.906283, 71.2762],
                             [0.0, 0.0, 0.0, 1.0]])
        intrinsic_matrix = np.array([[1446.165, 0, 320], [0, 1441.59, 256], [0, 0, 1]])

        src_matrix, ref_matrix = np.eye(4), np.eye(4)

        src_matrix[:3, :4] = np.dot(intrinsic_matrix, ref_pose[:3])
        ref_matrix[:3, :4] = np.dot(intrinsic_matrix, ref_pose[:3])
        src_matrix = torch.tensor(src_matrix).unsqueeze(0).repeat(batch_size, 1, 1)
        ref_matrix = torch.tensor(ref_matrix).unsqueeze(0).repeat(batch_size, 1, 1)

        depth_min, depth_interval, num_depth = 425, 2.5, 64
        depth_max = depth_min + num_depth * depth_interval
        depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
        depth_values = torch.tensor(depth_values).unsqueeze(0).repeat(batch_size, 1)
        depth_values = depth_values.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)

        src_vol = torch.ones((batch_size, num_depth, height, width)) #torch.rand(batch_size, num_depth, height, width)
        src_vol = F.log_softmax(src_vol, dim=1)

        warped_vol = resample_vol(src_vol.float().cuda(), src_matrix.float().cuda(), ref_matrix.float().cuda(),
                                  depth_values.float().cuda())
        print(warped_vol.size())
        check = torch.mean(torch.exp(warped_vol) - 1.0 / num_depth) < 0.00001
        print(torch.mean(torch.exp(warped_vol) - 1.0 / num_depth))
        self.assertTrue(check)

    def test_volume2(self):
        batch_size = 1
        height, width = 480, 752

        src_depth = np.load('/home/khangtg/Documents/lab/code/defusr/test_warping/gt_depth1.npy')
        ref_depth = np.load('/home/khangtg/Documents/lab/code/defusr/test_warping/gt_depth2.npy')
        src_pose = np.loadtxt('/home/khangtg/Documents/lab/code/defusr/test_warping/pose1.txt')
        ref_pose = np.loadtxt('/home/khangtg/Documents/lab/code/defusr/test_warping/pose2.txt')
        K = np.loadtxt('/home/khangtg/Documents/lab/code/defusr/test_warping/intrinsic.txt')

        src_matrix, ref_matrix = np.eye(4), np.eye(4)
        src_matrix[:3, :4] = np.dot(K, src_pose[:3])
        ref_matrix[:3, :4] = np.dot(K, ref_pose[:3])
        src_matrix = torch.tensor(src_matrix).unsqueeze(0).repeat(batch_size, 1, 1)
        ref_matrix = torch.tensor(ref_matrix).unsqueeze(0).repeat(batch_size, 1, 1)

        depth_min, depth_max, num_depth = 0.0, 6.0, 128
        depth_interval = (depth_max - depth_min) / num_depth
        depth_values = np.arange(depth_min+depth_interval/2, depth_max, depth_interval, dtype=np.float32)
        depth_values = torch.tensor(depth_values).unsqueeze(0).repeat(batch_size, 1)
        depth_values = depth_values.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)
        # print(depth_values[0][2])

        # build src_cost_volume
        src_vol = torch.zeros((batch_size, num_depth, height, width)) + 0.0001 # [B, D, H, W]
        depth_ind = ((src_depth - depth_min) / depth_interval) # [H, W]
        depth_ind = torch.tensor(depth_ind, dtype=torch.int64)
        height_ind, width_ind = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        src_vol[:, depth_ind, height_ind, width_ind] = 1
        src_vol = F.normalize(src_vol, p=1, dim=1)
        src_ind = torch.argmax(src_vol, dim=1).squeeze(0).cpu()
        src_vol = torch.log(src_vol)
        warped_vol = resample_vol(src_vol.float().cuda(), src_matrix.float().cuda(), ref_matrix.float().cuda(),
                                  depth_values.float().cuda(), prev_depth_values=None)
        warped_vol = torch.exp(warped_vol)
        ind = torch.argmax(warped_vol, dim=1)
        ind = ind.squeeze(0).float().cpu().numpy()
        gt_ind = (ref_depth - depth_min) / depth_interval
        mask = ref_depth > 0
        print(ind[mask][1000:1010], gt_ind[mask][1000:1010])
        diff = np.mean(np.abs(ind[mask] - gt_ind[mask]))
        print(diff)
        self.assertTrue((diff < 1e-3))


if __name__ == '__main__':
    unittest.main()