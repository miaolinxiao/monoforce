import torch
from torch import nn
# from torchvision.models.resnet import resnet18
import torchvision.models as models

# 假定你已经有以下工具函数和模块（例如 gen_dx_bx, cumsum_trick, QuickCumsum）
from .utils import gen_dx_bx, cumsum_trick, QuickCumsum

########################################
# 通用辅助模块
########################################

class ScaledTanh(nn.Module):
    def __init__(self, min_val=-1., max_val=1.):
        super(ScaledTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * (torch.tanh(x) + 1) / 2

########################################
# CamEncode 模块（保持不变）
########################################

class CamEncode(nn.Module):
    def __init__(self, D, C):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        # 使用 torchvision 的 efficientnet_v2_s 模型，加载预训练权重（若可用）
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if hasattr(models, "EfficientNet_V2_S_Weights") else None
        self.trunk = models.efficientnet_v2_s(weights=weights)
        # 去掉分类器，仅保留特征提取部分
        self.trunk.classifier = nn.Identity()
        # 修改 Up 层输入通道数：假设 reduction_5 与 reduction_4 分别对应 1280 与 160
        self.up1 = Up(1280 + 160, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def get_eff_depth(self, x):
        endpoints = {}
        prev_x = None
        for idx, layer in enumerate(self.trunk.features):
            if idx == 0:
                x = layer(x)
                prev_x = x
            else:
                x = layer(x)
                if prev_x.shape[-2:] != x.shape[-2:]:
                    endpoints[f'reduction_{len(endpoints)+1}'] = prev_x
                prev_x = x
        endpoints[f'reduction_{len(endpoints)+1}'] = x
        if 'reduction_5' in endpoints and 'reduction_4' in endpoints:
            x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        else:
            keys = sorted(endpoints.keys())
            if len(keys) >= 2:
                x = self.up1(endpoints[keys[-1]], endpoints[keys[-2]])
            else:
                x = x
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x

########################################
# Up 模块（原始代码中的上采样模块）
########################################

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

########################################
# Transformer-Based BEV 编码器
########################################

class TransformerBEVEncode(nn.Module):
    def __init__(self, inC, embed_dim, num_transformer_layers, outC):
        """
        inC: voxel_pooling 输出的通道数（例如 camC）
        embed_dim: Transformer 的嵌入维度
        num_transformer_layers: Transformer Encoder 层数
        outC: 最终输出的通道数（例如 terrain 对应的通道数）
        """
        super(TransformerBEVEncode, self).__init__()
        # 先用 1x1 卷积将 inC 映射到 embed_dim
        self.embed_conv = nn.Conv2d(inC, embed_dim, kernel_size=1)
        # 这里使用 learnable 位置编码，后续根据实际输入的空间尺寸动态生成
        self.register_parameter("pos_embed", None)
        # 构建 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # 输出 head 分支
        self.head_geom = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, outC, kernel_size=1, padding=0),
            ScaledTanh(-1, 1)
        )
        self.head_diff = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, outC, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.head_friction = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, outC, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, inC, H, W)
        B, _, H, W = x.shape
        x = self.embed_conv(x)  # (B, embed_dim, H, W)
        # Flatten空间维度，形状变为 (B, H*W, embed_dim)
        x_flat = x.flatten(2).transpose(1, 2)
        seq_len = x_flat.size(1)
        # 如果没有预先定义位置编码，动态生成 learnable 的位置 embedding
        if self.pos_embed is None or self.pos_embed.shape[1] < seq_len:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, x_flat.size(-1)).to(x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x_flat = x_flat + self.pos_embed[:, :seq_len, :]
        # Transformer 需要 (seq_len, B, embed_dim) 的输入格式
        x_flat = x_flat.transpose(0, 1)
        x_trans = self.transformer(x_flat)
        # 恢复形状到 (B, embed_dim, H, W)
        x_trans = x_trans.transpose(0, 1)  # (B, seq_len, embed_dim)
        x_trans = x_trans.transpose(1, 2).view(B, -1, H, W)
        # 通过各个 head 分支生成输出
        x_geom = self.head_geom(x_trans)
        x_diff = self.head_diff(x_trans)
        x_friction = self.head_friction(x_trans)
        x_terrain = x_geom - x_diff
        out = {
            'geom': x_geom,
            'terrain': x_terrain,
            'diff': x_diff,
            'friction': x_friction
        }
        return out

########################################
# 基于 Transformer 的 LiftSplatShoot 网络
########################################

class LiftSplatShoot_Transformer(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1, transformer_embed_dim=None, num_transformer_layers=4):
        """
        grid_conf: 包含 xbound, ybound, zbound, dbound 等参数
        data_aug_conf: 包含 final_dim 等配置
        outC: 最终输出通道数（例如 terrain 的通道数）
        transformer_embed_dim: Transformer 的嵌入维度，默认设为 camC * depth_bins
        num_transformer_layers: Transformer Encoder 层数
        """
        super(LiftSplatShoot_Transformer, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC)
        
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        num_depth_bins = int(self.nx[2].item())
        if transformer_embed_dim is None:
            transformer_embed_dim = self.camC * num_depth_bins
        self.transformer_bevencode = TransformerBEVEncode(inC=self.camC,
                                                          embed_dim=transformer_embed_dim,
                                                          num_transformer_layers=num_transformer_layers,
                                                          outC=outC)
        self.use_quickcumsum = True

    def create_frustum(self):
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """计算点云在 ego 坐标系中的位置"""
        B, N, _ = trans.shape
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        """
        输入 x: (B, N, C, imH, imW)
        输出: (B, N, D, H', W', camC)
        """
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def voxel_pooling(self, geom_feats, x):
        """
        将 voxel_pooling 应用于每个像素，将多帧特征汇聚到 BEV 网格中
        输入 x: (B, N, D, H, W, C)
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        x: (B, N, C, imH, imW)
        rots, trans, intrins, post_rots, post_trans 均为相应的摄像头参数
        """
        bev_feats = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        out = self.transformer_bevencode(bev_feats)
        return out
    
    def from_pretrained(self, modelf):
        if not modelf:
            return self
        print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        model_dict = self.state_dict()
        pretrained_model = torch.load(modelf)
        model_dict.update(pretrained_model)
        self.load_state_dict(model_dict, strict=False)
        return self
    
    # def _resize_pos_embed(old_pos_embed, new_seq_len):
    #     import math
    #     import torch.nn.functional as F
    #     """
    #     old_pos_embed: shape (1, old_seq_len, D)
    #     new_seq_len: 新的序列长度，假设新旧空间布局基本为网格（即旧的seq_len和新的seq_len均可以开方）
    #     若不能整除，需要按照具体情况调整策略。
    #     """
    #     # 假设旧的序列长度为 old_size^2，新序列长度为 new_size^2
    #     _, old_seq_len, D = old_pos_embed.shape
    #     old_size = int(math.sqrt(old_seq_len))
    #     new_size = int(math.sqrt(new_seq_len))
    #     if old_size * old_size != old_seq_len or new_size * new_size != new_seq_len:
    #         raise ValueError("旧或新的序列长度不符合正方形网格假设，请自定义重采样流程。")
    #     # 重塑为 (1, D, old_size, old_size)
    #     old_pos_embed = old_pos_embed.reshape(1, old_size, old_size, D).permute(0, 3, 1, 2)
    #     new_pos_embed = F.interpolate(old_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
    #     new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, D)
    #     return new_pos_embed
    # def from_pretrained(self, modelf):
    #     if not modelf:
    #         return self
    #     print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
    #     model_dict = self.state_dict()
    #     pretrained_model = torch.load(modelf)
    #     new_state_dict = {}
    #     for k, v in pretrained_model.items():
    #         if k == "transformer_bevencode.pos_embed":
    #             # 假设当前输入对应的序列长度
    #             # 注意：需要根据实际情况确定 new_seq_len，这里假设能通过模型或配置获取
    #             # 例如：new_seq_len = 当前 BEV 特征的 H * W
    #             new_seq_len = model_dict[k].shape[1] if k in model_dict else v.shape[1]
    #             if v.shape[1] != new_seq_len:
    #                 print(f"Resizing pos_embed from {v.shape[1]} to {new_seq_len}")
    #                 v = self._resize_pos_embed(v, new_seq_len)
    #         new_state_dict[k] = v
    #     model_dict.update(new_state_dict)
    #     self.load_state_dict(model_dict)
    #     return self