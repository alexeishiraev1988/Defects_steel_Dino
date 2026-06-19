import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM   = 768
NUM_CLASSES = 5
PATCH_H     = 16   # 224 // 14
PATCH_W     = 100  # 1400 // 14


class SegHeadDPT(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                 patch_h=PATCH_H, patch_w=PATCH_W, n_layers=4):
        super().__init__()
        self.patch_h  = patch_h
        self.patch_w  = patch_w
        self.n_layers = n_layers
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, 256, 1),
                          nn.BatchNorm2d(256), nn.GELU())
            for _ in range(n_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * n_layers, 512, 1), nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, features):
        maps = []
        for i, f in enumerate(features):
            B, N, C = f.shape
            x = f.reshape(B, self.patch_h, self.patch_w, C).permute(0, 3, 1, 2)
            maps.append(self.proj[i](x))
        x = self.fuse(torch.cat(maps, dim=1))
        return self.head(self.up2(self.up1(x)))
