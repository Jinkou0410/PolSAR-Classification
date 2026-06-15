import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models



class CoordinateAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_fused = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()

        # 1. 坐标信息嵌入
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        fused_map = self.act(x_h + x_w.permute(0, 1, 3, 2))
        spatial_attn = self.conv_fused(fused_map)

        grid_h = torch.linspace(-1, 1, H, device=x.device)
        grid_w = torch.linspace(-1, 1, W, device=x.device)
        mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        center_mask = torch.exp(-(mesh_h ** 2 + mesh_w ** 2) * 2.0)
        center_mask = center_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        final_attn = (spatial_attn * center_mask).sigmoid()

        return x * final_attn

class PhysicalMappingLayer(nn.Module):
    def __init__(self, in_channels=3, mid_channels=128):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding=1),
        )

    def forward(self, x): return self.mapping(x)


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v=None, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_k, dim_q)
        self.v_proj = nn.Linear(dim_v if dim_v else dim_k, dim_q)
        self.proj = nn.Linear(dim_q, dim_q)

    def forward(self, x_q, x_kv):
        B, Nq, _ = x_q.shape
        Nk = x_kv.shape[1]
        q = self.q_proj(x_q).reshape(B, Nq, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, Nq, -1)
        return self.proj(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=15, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return self.proj(x).flatten(2).transpose(1, 2)


class CrossVITBlock(nn.Module):
    def __init__(self, dim_L, dim_S, num_heads=8):
        super().__init__()
        self.norm1_L, self.norm1_S = nn.LayerNorm(dim_L), nn.LayerNorm(dim_S)
        self.attn_L = CrossAttention(dim_L, dim_S, num_heads=num_heads)
        self.attn_S = CrossAttention(dim_S, dim_L, num_heads=num_heads)
        self.mlp_L = nn.Sequential(nn.Linear(dim_L, dim_L * 4), nn.GELU(), nn.Linear(dim_L * 4, dim_L))
        self.mlp_S = nn.Sequential(nn.Linear(dim_S, dim_S * 4), nn.GELU(), nn.Linear(dim_S * 4, dim_S))

    def forward(self, x_L, x_S):
        x_L = x_L + self.attn_L(self.norm1_L(x_L), self.norm1_S(x_S))
        x_S = x_S + self.attn_S(self.norm1_S(x_S), self.norm1_L(x_L))
        x_L = x_L + self.mlp_L(self.norm1_L(x_L))
        x_S = x_S + self.mlp_S(self.norm1_S(x_S))
        return x_L, x_S



class PhyFeatureExtractor(nn.Module):
    def __init__(self, in_channels_main=9, in_channels_phy=3, img_size=15,
                 use_dpi=True, use_ccam=False, patch_size_L=15, patch_size_S=5):
        super().__init__()
        self.use_dpi = use_dpi
        self.use_ccam = use_ccam
        self.phy_encoder = PhysicalMappingLayer(in_channels=in_channels_phy, mid_channels=128)
        self.patch_embed_L = PatchEmbed(img_size=img_size, patch_size=patch_size_L,
                                        in_channels=in_channels_main, embed_dim=256)
        self.patch_embed_S = PatchEmbed(img_size=img_size, patch_size=patch_size_S,
                                        in_channels=in_channels_main, embed_dim=128)

        self.cls_token_L = nn.Parameter(torch.zeros(1, 1, 256))
        self.cls_token_S = nn.Parameter(torch.zeros(1, 1, 128))

        self.blocks_L = nn.ModuleList([CrossVITBlock(256, 128, 8) for _ in range(4)])
        self.decoder_L = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=patch_size_L, stride=patch_size_L),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.decoder_S = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=patch_size_S, stride=patch_size_S),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.phy_inject2 = nn.Conv2d(256, 128, 1)
        self.phy_inject3 = nn.Conv2d(256, 256, 1)

        self.phy_guide1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 64), nn.Sigmoid()
        )
        self.phy_guide2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128), nn.Sigmoid()
        )

        self.final_fuse = nn.Sequential(
            nn.Conv2d(128+64, 64, 3, padding=1),
            CoordinateAttention(64) if use_dpi else nn.Identity(),
            nn.ReLU()
        )

    def forward(self, x_main, x_phy):

        B = x_main.shape[0]

        phy_feat = self.phy_encoder(x_phy)

        x_L = torch.cat((self.cls_token_L.expand(B, -1, -1), self.patch_embed_L(x_main)), dim=1)
        x_S = torch.cat((self.cls_token_S.expand(B, -1, -1), self.patch_embed_S(x_main)), dim=1)

        for block in self.blocks_L:
            x_L, x_S = block(x_L, x_S)

        grid_L = int(np.sqrt(x_L.shape[1] - 1))
        grid_S = int(np.sqrt(x_S.shape[1] - 1))
        feat_L = x_L[:, 1:].reshape(B, 256, grid_L, grid_L)
        feat_S = x_S[:, 1:].reshape(B, 128, grid_S, grid_S)

        feat_L = F.interpolate(self.phy_inject3(phy_feat), size=feat_L.shape[2:]) + feat_L
        feat_S = F.interpolate(self.phy_inject2(phy_feat), size=feat_S.shape[2:]) + feat_S

        out_L = self.decoder_L(feat_L)
        out_S = self.decoder_S(feat_S)
        if out_L.shape[2:] != out_S.shape[2:]:
            out_S = F.interpolate(out_S, size=out_L.shape[2:], mode='bilinear')

        out_L = out_L * F.interpolate(self.phy_guide2(phy_feat).view(B, 128, 1, 1), size=out_L.shape[2:])
        out_S = out_S * F.interpolate(self.phy_guide1(phy_feat).view(B, 64, 1, 1), size=out_S.shape[2:])

        fused = self.final_fuse(torch.cat([out_L, out_S], dim=1))
        return fused

class PolSARPhyNet(nn.Module):
    def __init__(self, num_classes=4, use_smpc=True, use_dpi=True, use_ccam=False, patch_size=15):
        super().__init__()
        self.use_smpc = use_smpc
        self.use_dpi = use_dpi
        self.use_ccam = use_ccam
        self.backbone = PhyFeatureExtractor(
            in_channels_main=9,
            in_channels_phy=3,
            img_size=patch_size,
            use_dpi=use_dpi,
            use_ccam=use_ccam,
            patch_size_L=15,
            patch_size_S=5
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

        if self.use_smpc or self.use_dpi:
            self.phy_recon = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 1)
            )

    def forward(self, x_main, x_phy, mode=None):
        if mode == 'project':
            feat = F.adaptive_avg_pool2d(x_main, 1).view(x_main.size(0), -1)
            return feat, None

        fused_feat = self.backbone(x_main, x_phy)

        logits = self.classifier(fused_feat)
        phy_pred = None
        if self.use_smpc or self.use_dpi:
            phy_pred = self.phy_recon(fused_feat)

        return logits, phy_pred

