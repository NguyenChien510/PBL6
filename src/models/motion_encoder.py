"""
Motion Encoder (S3D with Separable Convolutions & Gating Mechanism) for MFGF TVPR
Processes L2=16 continuous frames.
Separates 3D Conv into 2D spatial [1, k, k] and 1D temporal [k, 1, 1].
Implements Gating Mechanism:
    y_i = sigmoid(sigma * y^p_i + b) * y^p_i
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv3d(nn.Module):
    """
    Spatio-Temporal Separable 3D Convolution:
      Step 1: 2D Spatial Conv with kernel (1, k, k)
      Step 2: 1D Temporal Conv with kernel (k, 1, 1)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        # Intermediate channel dimension
        mid_channels = out_channels

        # Spatial convolution (1, k, k)
        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=bias
        )
        self.bn_s = nn.BatchNorm3d(mid_channels)
        self.relu_s = nn.ReLU(inplace=True)

        # Temporal convolution (k, 1, 1)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=bias
        )
        self.bn_t = nn.BatchNorm3d(out_channels)
        self.relu_t = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.relu_s(self.bn_s(self.conv_s(x)))
        x = self.relu_t(self.bn_t(self.conv_t(x)))
        return x


class S3DInceptionBlock(nn.Module):
    """
    Multi-Branch Inception Block with Separable 3D Convolutions.
    """
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        pool_proj: int
    ):
        super().__init__()
        # Branch 1: 1x1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv3d(in_channels, out_1x1, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1x1 conv -> 3x3x3 separable conv
        self.b2 = nn.Sequential(
            nn.Conv3d(in_channels, red_3x3, kernel_size=1, bias=False),
            nn.BatchNorm3d(red_3x3),
            nn.ReLU(inplace=True),
            SepConv3d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # Branch 3: 1x1x1 conv -> 3x3x3 separable conv (emulating 5x5)
        self.b3 = nn.Sequential(
            nn.Conv3d(in_channels, red_5x5, kernel_size=1, bias=False),
            nn.BatchNorm3d(red_5x5),
            nn.ReLU(inplace=True),
            SepConv3d(red_5x5, out_5x5, kernel_size=3, padding=1)
        )

        # Branch 4: 3x3x3 MaxPool -> 1x1x1 conv
        self.b4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.Conv3d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm3d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class MotionEncoder(nn.Module):
    """
    S3D Motion Encoder for L2=16 consecutive frames.
    Implements:
      1. Stem with separable convolutions
      2. Inception blocks for multi-scale motion modeling
      3. Spatio-temporal Average Pooling -> y^p_i
      4. Gating Mechanism:
            y_i = sigmoid(sigma * y^p_i + b) * y^p_i
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 16, # L2 = 16
        embed_dim: int = 768
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Stem: Initial downsampling layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.conv2 = nn.Sequential(
            SepConv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        # Inception Stages
        # Stage 1: Inception Block (out = 64 + 128 + 32 + 32 = 256)
        self.inc1 = S3DInceptionBlock(128, out_1x1=64, red_3x3=64, out_3x3=128, red_5x5=16, out_5x5=32, pool_proj=32)

        # Stage 2: Inception Block (out = 128 + 192 + 64 + 64 = 448)
        self.inc2 = S3DInceptionBlock(256, out_1x1=128, red_3x3=96, out_3x3=192, red_5x5=32, out_5x5=64, pool_proj=64)

        # Final projection to embed_dim before pooling
        self.proj = nn.Sequential(
            nn.Conv3d(448, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Global Spatio-Temporal Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Learnable Gating Parameters: sigma and b
        # y_i = sigmoid(sigma * y^p_i + b) * y^p_i
        self.sigma = nn.Parameter(torch.ones(embed_dim))
        self.b = nn.Parameter(torch.zeros(embed_dim))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, L2, H, W]
        Returns:
            f_mot: Gated motion feature of shape [B, embed_dim]
        """
        b, c, t, h, w = x.shape

        # S3D Forward Pass
        feat = self.conv1(x) # [B, 64, T, H/4, W/4]
        feat = self.conv2(feat) # [B, 128, T/2, H/8, W/8]
        feat = self.inc1(feat) # [B, 256, ...]
        feat = self.inc2(feat) # [B, 448, ...]
        feat = self.proj(feat) # [B, embed_dim, ...]

        # Spatio-temporal Average Pooling -> y^p
        y_p = self.global_pool(feat).view(b, self.embed_dim) # [B, embed_dim]

        # Gating Mechanism: y_i = sigmoid(sigma * y^p_i + b) * y^p_i
        # sigma and b are channel-wise learnable vectors
        gate = torch.sigmoid(self.sigma * y_p + self.b)
        f_mot = gate * y_p # Element-wise multiplication

        f_mot = self.norm(f_mot)
        return f_mot
