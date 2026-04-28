import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GlobalSemanticAwareWeight(nn.Module):
    def __init__(self, in_channels=256, d_model=128):
        super().__init__()
        # 1. 先做维度对齐：把输入256维映射到Mamba要求的d_model=128
        self.input_proj_1x = nn.Linear(in_channels, d_model)
        self.input_proj_2x = nn.Linear(in_channels//2, d_model//2)
        self.input_proj_4x = nn.Linear(in_channels//4, d_model//4)
        
        # 多尺度下采样层（1x → 2x → 4x）
        self.downsample_2x = nn.Linear(in_channels, in_channels//2)
        self.downsample_4x = nn.Linear(in_channels//2, in_channels//4)
        
        # 轻量Mamba模块（旧版兼容，维度完全对齐）
        self.mamba_1x = Mamba(
            d_model=d_model, d_state=16, d_conv=4, expand=2
        )
        self.mamba_2x = Mamba(
            d_model=d_model//2, d_state=16, d_conv=4, expand=2
        )
        self.mamba_4x = Mamba(
            d_model=d_model//4, d_state=16, d_conv=4, expand=2
        )
        
        # 多尺度特征融合 + 权重预测头
        self.weight_head = nn.Sequential(
            nn.Linear(d_model + d_model//2 + d_model//4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, voxel_feat):
        """
        voxel_feat: [V, C] 体素特征（x_conv4.features，256维）
        return: [V,] 全局语义权重
        """
        # 1. 多尺度特征提取
        feat_1x = voxel_feat  # [V, 256]
        feat_2x = self.downsample_2x(feat_1x)  # [V, 128]
        feat_4x = self.downsample_4x(feat_2x)  # [V, 64]

        # 2. 维度对齐：映射到Mamba要求的d_model
        feat_1x_proj = self.input_proj_1x(feat_1x)  # [V, 128]
        feat_2x_proj = self.input_proj_2x(feat_2x)  # [V, 64]
        feat_4x_proj = self.input_proj_4x(feat_4x)  # [V, 32]

        # 3. Mamba全局建模（维度完全匹配，无矩阵乘法错误）
        feat_1x_mamba = self.mamba_1x(feat_1x_proj.unsqueeze(0)).squeeze(0)  # [V, 128]
        feat_2x_mamba = self.mamba_2x(feat_2x_proj.unsqueeze(0)).squeeze(0)  # [V, 64]
        feat_4x_mamba = self.mamba_4x(feat_4x_proj.unsqueeze(0)).squeeze(0)  # [V, 32]

        # 4. 多尺度特征融合
        fused_feat = torch.cat([feat_1x_mamba, feat_2x_mamba, feat_4x_mamba], dim=1)  # [V, 224]

        # 5. 预测全局语义权重
        w_global = self.weight_head(fused_feat).squeeze(1)  # [V,]
        return w_global