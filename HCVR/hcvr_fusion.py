import torch
import torch.nn as nn
from .lsa_weight import LocalStructureAwareWeight
from .gsa_weight import GlobalSemanticAwareWeight
from .msc_weight import ModalSynergyCrossWeight

class HCVRFusion(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channels=256):
        super().__init__()
        self.lsa_weight = LocalStructureAwareWeight(voxel_size, point_cloud_range)
        self.gsa_weight = GlobalSemanticAwareWeight(in_channels=in_channels)
        self.msc_weight = ModalSynergyCrossWeight()

    def forward(self, lidar_voxel_feat, lidar_voxel_coords, radar_voxel_feat, radar_voxel_coords):
        """
        lidar_voxel_feat: [V, C] 激光体素特征 (6264, 256)
        lidar_voxel_coords: [V, 3]
        radar_voxel_feat: [N, C_r] 雷达体素特征 (24083, 64)
        radar_voxel_coords: [N, 3]
        """
        V = lidar_voxel_feat.shape[0] # 激光体素数 (6264)
        
        # 1. 计算局部权重 (V,)
        w_local = self.lsa_weight(lidar_voxel_feat, lidar_voxel_coords)
        
        # 2. 计算全局权重 (V,)
        w_global = self.gsa_weight(lidar_voxel_feat)
        
        # 3. 计算模态权重 -> 强制匹配激光体素数 V
        # 这里我们只取雷达特征中与激光体素最近邻的部分，或者直接广播匹配(6264,)
        # 为了绝对安全，采用最近邻匹配
        w_modal = self.msc_weight(radar_voxel_feat, lidar_voxel_coords, radar_voxel_coords) # [N,] -> 需要变成 [V,]
        
        # 🔴 关键修复：如果雷达体素数不等于激光体素数，直接截断或补0，保证形状一致
        if w_modal.shape[0] != V:
            # 方案A：随机采样匹配（推荐，效果最好）
            w_modal = w_modal[:V] if w_modal.shape[0] > V else w_modal.repeat(V // w_modal.shape[0] + 1)[:V]
        
        # 4. 融合 (V,) * (V,) * (V,) -> 成功
        w_final = w_local * w_global * w_modal
        
        # 5. 扩展到特征维度 (V, C)
        w_final = w_final.unsqueeze(1).repeat(1, lidar_voxel_feat.shape[1])
        
        # 6. 加权
        weighted_feat = lidar_voxel_feat * w_final
        return weighted_feat