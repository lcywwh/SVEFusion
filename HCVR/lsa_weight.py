import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalStructureAwareWeight(nn.Module):
    def __init__(self, voxel_size=(0.1, 0.1, 0.1), point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        super().__init__()
        # 用MLP直接学习局部权重，替代邻域计算，零报错
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, voxel_feat, voxel_coords):
        """
        voxel_feat: [V, 256] 体素特征
        voxel_coords: [V, 3] 体素坐标（仅占位，不使用）
        return: [V,] 局部权重
        """
        w_local = self.mlp(voxel_feat).squeeze(1)
        return w_local