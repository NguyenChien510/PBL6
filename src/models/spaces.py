"""
Common Space and Dual-Distilled (D2) Space Projections for MFGF TVPR
Implements:
  - FeatureConvertor: Sequential(Linear -> BatchNorm1d -> Sigmoid) mapping to D_tips dimension.
  - CommonSpaceProjector: Maps representations into 256-D Common Space with L2 normalization.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureConvertor(nn.Module):
    """
    FeatureConvertor:
      A sequential block of (Linear -> BatchNorm1d -> Sigmoid)
      mapping Common Space representations into D^2 Space (D_tips dimension).
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 1000):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, in_dim]
        Returns:
            Tensor of shape [B, out_dim] with values in (0, 1)
        """
        x = self.fc(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x


class CommonSpaceProjector(nn.Module):
    """
    Projector that maps multimodal representations (f_Text and f_ME)
    into the 256-D Common Space with L2 normalization.
    """
    def __init__(self, in_dim: int = 768, common_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(inplace=True),
            nn.Linear(common_dim, common_dim)
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, in_dim]
        Returns:
            Projected common space tensor [B, common_dim]
        """
        out = self.proj(x)
        if normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out
