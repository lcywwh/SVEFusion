import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalSynergyCrossWeight(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, radar_voxel_feat, lidar_voxel_coords, radar_voxel_coords):
        """
        radar_voxel_feat: [N, 64] 雷达特征
        lidar_voxel_coords: [V, 3] 激光坐标 (目标坐标)
        radar_voxel_coords: [N, 3] 雷达坐标
        return: [V,] 模态权重
        """
        # 1. 计算雷达显著性
        doppler = torch.abs(radar_voxel_feat[:, 0])
        power = radar_voxel_feat[:, 1]
        radar_saliency = self.alpha * doppler + self.beta * power # [N,]

        # 2. 最近邻匹配：为每个激光体素找最近的雷达体素
        dist = torch.cdist(lidar_voxel_coords.float(), radar_voxel_coords.float()) # [V, N]
        min_dist_idx = dist.argmin(dim=-1) # [V,] 每个激光体素对应的最近雷达索引
        
        # 3. 生成激光体素对应的模态权重
        w_modal = radar_saliency[min_dist_idx] # [V,] 🔴 关键：输出维度必须是 [V]
        w_modal = torch.sigmoid(w_modal)
        return w_modal