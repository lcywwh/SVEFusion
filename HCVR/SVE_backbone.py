from .hcvr_fusion import HCVRFusion

from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils, box_utils

from spconv.pytorch import functional as Fsp
from typing import List
import numpy as np


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SVEBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(64),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.conv2 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.conv3 = spconv.SparseSequential(
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.conv4 = spconv.SparseSequential(
            block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.upconv1 = spconv.SparseSequential(
            block(256, 256, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv'),
        )
        self.upconv2 = spconv.SparseSequential(
            block(256, 128, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv'),
        )
        self.upconv3 = spconv.SparseSequential(
            block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv'),
        )

        self.convupconv1 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 0, 0), indice_key='spconv6', conv_type='spconv'),
        )
        self.convupconv2 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 0, 0), indice_key='spconv7', conv_type='spconv'),
        )
        self.convupconv3 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 0, 0), indice_key='spconv8', conv_type='spconv'),
        )
            
        last_pad = (0, 0, 0)
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
        self.num_point_features = 64
        
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 128,
            'x_conv3': 256,
            'x_conv4': 256
        }

        # HCVR 层级体素重加权
        self.hcvr_fusion = HCVRFusion(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            in_channels=256
        )
        
    def _indice_to_scalar(self, indices: torch.Tensor, shape: List[int]):
        assert indices.shape[1] == len(shape)
        stride = to_stride(np.array(shape, dtype=np.int64))
        scalar_inds = indices[:, -1].clone()
        for i in range(len(shape) - 1):
            scalar_inds += stride[i] * indices[:, i]
        return scalar_inds.contiguous()

    def forward(self, batch_dict):
        self.voxel_size = batch_dict['voxel_size']
        batch_size = batch_dict['batch_size']

        lidar_features = batch_dict['lidar_features']
        radar_features = batch_dict['radar_features']
        lidar_coords = batch_dict['lidar_voxel_coords']
        radar_coords = batch_dict['radar_voxel_coords']

        lidar_sp_tensor = spconv.SparseConvTensor(
            features=lidar_features,
            indices=lidar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        radar_sp_tensor = spconv.SparseConvTensor(
            features=radar_features,
            indices=radar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        fusion_sp_tensor = Fsp.sparse_add(lidar_sp_tensor, radar_sp_tensor)
        input_sp_tensor = fusion_sp_tensor

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # ==================== HCVR 体素增强 ====================
        lidar_voxel_coords = lidar_coords[:, 1:4]
        radar_voxel_coords = radar_coords[:, 1:4]

        hcvr_weighted_feat = self.hcvr_fusion(
            lidar_voxel_feat=x_conv4.features,
            lidar_voxel_coords=lidar_voxel_coords,
            radar_voxel_feat=radar_features,
            radar_voxel_coords=radar_voxel_coords
        )
        x_conv4 = replace_feature(x_conv4, hcvr_weighted_feat)
        # ========================================================

        x_upconv1 = self.upconv1(x_conv4)
        x_conv3 = replace_feature(x_conv3, x_conv3.features * x_upconv1.features)

        x_upconv2 = self.upconv2(x_conv3)
        x_conv2 = replace_feature(x_conv2, x_conv2.features * x_upconv2.features)

        x_upconv3 = self.upconv3(x_conv2)
        x_conv1 = replace_feature(x_conv1, x_conv1.features * x_upconv3.features)

        x = replace_feature(x, x.features * x_conv1.features)

        conv_x_conv1 = self.convupconv1(x)
        conv_x_conv2 = self.convupconv2(conv_x_conv1)
        conv_x_conv3 = self.convupconv3(conv_x_conv2)
        out = self.conv_out(conv_x_conv3)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 1,
            'x_indices': x.indices,
            'x_conv1_indices': x_conv1.indices,
            'x_conv2_indices': x_conv2.indices,
            'x_conv3_indices': x_conv3.indices,
            'x_conv4_indices': x_conv4.indices,
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict

    def get_loss(self, batch_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        loss = 0.0
        tb_dict['loss_hcvr'] = 0.0
        return loss, tb_dict