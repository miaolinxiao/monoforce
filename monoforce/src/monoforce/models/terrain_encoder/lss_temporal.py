"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from .lss import LiftSplatShoot 

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        embed_dim: BEV 特征的通道数（即融合后每个空间位置的维度）
        num_heads: 多头注意力的头数
        """
        super(TemporalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        # query: (L, B, embed_dim)； key, value: (L_history, B, embed_dim)
        attn_out, _ = self.multihead_attn(query, key, value)
        out = self.norm(query + self.dropout(attn_out))
        return out

class LiftSplatShoot_Temporal(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC=1):
        super(LiftSplatShoot_Temporal, self).__init__(grid_conf, data_aug_conf, outC=outC)

        num_depth_bins = int(self.nx[2].item())
        embed_dim = self.camC * num_depth_bins

        # 初始化时序注意力模块
        self.temporal_attn = TemporalAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, bev_history):        
        
        bev_current = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        
        if bev_history is not None:
            # 将历史帧特征合并：历史帧 shape 为 (B, C_bev, H_bev, W_bev)
            # bev_history = torch.stack(bev_feats_list[:-1], dim=1)
            B, C_bev, H_bev, W_bev = bev_history.shape
            L = H_bev * W_bev
            # 将历史帧的空间维度 flatten
            bev_history_flat = bev_history.view(B, C_bev, L).permute(2, 0, 1)    # (L, B, C_bev)
            bev_current_flat = bev_current.view(B, C_bev, L).permute(2, 0, 1)    # (L, B, C_bev)

            # 使用时序注意力模块融合：当前帧 query 对历史帧 key/value 进行注意力加权
            bev_fused_flat = self.temporal_attn(bev_current_flat, bev_history_flat, bev_history_flat)
            bev_fused = bev_fused_flat.permute(1, 2, 0).view(B, C_bev, H_bev, W_bev)
            # 这样做的目的是让隐藏状态不再追踪之前时间步的梯度，只保留当前时间步的计算图，从而避免重复反向传播同一计算图。
            bev_fused = bev_fused.detach()
        else:
            bev_fused = bev_current
        # 将融合后的 BEV 特征送入 BEV 编码器，生成 terrain, geom, diff, friction 等输出
        out = self.bevencode(bev_fused)
        return out, bev_fused
    
    def from_pretrained(self, modelf):
        if not modelf:
            return self
        print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

        pretrained_model = torch.load(modelf)
        # 检查预训练权重中是否包含 temporal_attn 的权重
        if any(key.startswith('temporal_attn') for key in pretrained_model.keys()):
            # 如果存在 temporal_attn，则认为权重对应于 LiftSplatShoot_Temporal 类型，全部加载
            self.load_state_dict(pretrained_model)
        else:
            # 如果不存在 temporal_attn，则认为权重对应于 LiftSplatShoot 类型，
            # 只加载 LiftSplatShoot 部分权重，保留当前 temporal_attn 的随机初始化
            filtered_dict = {k: v for k, v in pretrained_model.items() if not k.startswith('temporal_attn')}
            self.load_state_dict(filtered_dict, strict=False)
        return self