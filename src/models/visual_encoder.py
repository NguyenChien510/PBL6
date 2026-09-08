"""
Visual Encoder (Modified Vision Transformer - ViT) for MFGF TVPR
Processes L1=4 random frames with additive Spatial and Temporal embeddings:
    e_{l,n} = e^O_{l,n} + E_{S_n} + E_{T_l}
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Splits image into patches and maps each patch to embed_dim.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Tensor of shape [B, num_patches, embed_dim]
        """
        x = self.proj(x) # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x


class VisualEncoder(nn.Module):
    """
    Modified Spatio-Temporal ViT:
      - Takes L1=4 frames: [B, C, L1, H, W]
      - Computes patch embeddings e^O_{l,n}
      - Adds Spatial Embedding (E_S) and Temporal Embedding (E_T):
            e_{l,n} = e^O_{l,n} + E_{S_n} + E_{T_l}
      - Passes tokens through Transformer Encoder blocks
      - Produces global visual representation f_vis in [B, embed_dim]
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_frames: int = 4, # L1 = 4
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches # 14x14 = 196

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Additive Spatial Embedding E_S: [1, 1, num_patches, embed_dim]
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, embed_dim) * 0.02)

        # Additive Temporal Embedding E_T: [1, num_frames, 1, embed_dim]
        self.temporal_embed = nn.Parameter(torch.randn(1, self.num_frames, 1, embed_dim) * 0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.spatial_embed, std=0.02)
        nn.init.normal_(self.temporal_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, L1, H, W]
        Returns:
            f_vis: Visual representation of shape [B, embed_dim]
        """
        # x shape: [B, C, L1, H, W]
        b, c, l, h, w = x.shape
        assert l == self.num_frames, f"Expected {self.num_frames} frames, got {l}"

        # Reshape to [B * L1, C, H, W] for patch embedding
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * l, c, h, w)
        patch_tokens = self.patch_embed(x) # [B * L1, N, D]

        # Reshape to [B, L1, N, D] where each element is e^O_{l,n}
        patch_tokens = patch_tokens.view(b, l, self.num_patches, self.embed_dim)

        # Spatio-Temporal Additive Embeddings:
        # e_{l,n} = e^O_{l,n} + E_{S_n} + E_{T_l}
        # spatial_embed is [1, 1, N, D] -> broadcasts across B and L
        # temporal_embed is [1, L1, 1, D] -> broadcasts across B and N
        tokens = patch_tokens + self.spatial_embed + self.temporal_embed # [B, L1, N, D]

        # Flatten frames and patches: [B, L1 * N, D]
        tokens = tokens.reshape(b, l * self.num_patches, self.embed_dim)

        # Prepend [CLS] token: [B, 1 + L1 * N, D]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.pos_drop(tokens)

        # Transformer Encoding
        encoded = self.transformer(tokens) # [B, 1 + L1 * N, D]
        encoded = self.norm(encoded)

        # Extract global representation from [CLS] token: f_vis
        f_vis = encoded[:, 0, :] # [B, embed_dim]

        return f_vis
