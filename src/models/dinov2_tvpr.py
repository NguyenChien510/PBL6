"""
DINOv2-TVPR: Text-to-Video Person Retrieval using Meta DINOv2 & VinAI PhoBERT
Architecture:
  - Video Encoder: Pre-trained Meta DINOv2 (ViT-S/14 or ViT-B/14) + Temporal Attention Aggregator
  - Text Encoder: Pre-trained VinAI PhoBERT v2 with adapter
  - Common Space: 256-D L2-normalized projections
  - Loss: Symmetric Cross-Modal InfoNCE with learnable temperature (CLIP style)
"""

from typing import Dict, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TemporalAttentionAggregator(nn.Module):
    """
    Aggregates frame-level visual embeddings across time using Multi-Head Self-Attention.
    Input: [B, L, D]
    Output: [B, D]
    """
    def __init__(self, embed_dim: int = 384, num_frames: int = 4, num_heads: int = 6, depth: int = 2, drop_rate: float = 0.1):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        b, l, d = x.shape
        if l == self.num_frames:
            x = x + self.temporal_pos_embed
        else:
            # Interpolate if frames differ
            pos = F.interpolate(self.temporal_pos_embed.transpose(1, 2), size=l, mode="linear", align_corners=False).transpose(1, 2)
            x = x + pos
            
        feat = self.transformer(x) # [B, L, D]
        feat = self.norm(feat)
        # Average pooling across temporal frames
        return feat.mean(dim=1) # [B, D]


class DINOv2TVPRModel(nn.Module):
    """
    End-to-End Foundation Model Pipeline for Text-to-Video Person Retrieval.
    Combines:
      - Meta DINOv2 (frozen base / fine-tuned top layers)
      - Temporal Transformer Aggregator
      - VinAI PhoBERT v2 (fine-tuned)
      - Learnable Temperature Scale (CLIP style)
    """
    def __init__(
        self,
        dinov2_name: str = "facebook/dinov2-small",
        phobert_name: str = "vinai/phobert-base-v2",
        visual_frames: int = 4,
        common_dim: int = 256,
        freeze_dinov2_backbone: bool = True,
        tune_dinov2_layers: int = 2, # fine-tune last 2 blocks of DINOv2
        freeze_dinov2: Optional[bool] = None,
    ):
        super().__init__()
        if freeze_dinov2 is not None:
            freeze_dinov2_backbone = freeze_dinov2
        self.visual_frames = visual_frames
        self.common_dim = common_dim

        # 1. Vision Backbone: Meta DINOv2
        print(f"[*] Loading DINOv2 Foundation Model: {dinov2_name}...", flush=True)
        try:
            self.dinov2 = AutoModel.from_pretrained(dinov2_name, use_safetensors=True)
        except Exception:
            self.dinov2 = AutoModel.from_pretrained(dinov2_name)
        dinov2_dim = self.dinov2.config.hidden_size # 384 for small, 768 for base

        # Freeze early layers of DINOv2 if requested
        if freeze_dinov2_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            # Unfreeze top N transformer layers
            if tune_dinov2_layers > 0 and hasattr(self.dinov2, "encoder"):
                for layer in self.dinov2.encoder.layer[-tune_dinov2_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"[+] DINOv2 base frozen, fine-tuning top {tune_dinov2_layers} layers.", flush=True)
            else:
                print(f"[+] DINOv2 completely frozen as feature extractor.", flush=True)

        # 2. Temporal Aggregator for Video
        self.temporal_aggregator = TemporalAttentionAggregator(
            embed_dim=dinov2_dim,
            num_frames=visual_frames,
            num_heads=6 if dinov2_dim % 6 == 0 else 8,
            depth=2
        )

        # 3. Text Backbone: PhoBERT v2
        print(f"[*] Loading PhoBERT v2 Model: {phobert_name}...", flush=True)
        try:
            self.phobert = AutoModel.from_pretrained(phobert_name, use_safetensors=True)
        except Exception:
            self.phobert = AutoModel.from_pretrained(phobert_name)
        phobert_dim = self.phobert.config.hidden_size # 768

        # Text adapter to match dimensions if needed
        if phobert_dim != dinov2_dim:
            self.text_adapter = nn.Sequential(
                nn.Linear(phobert_dim, dinov2_dim),
                nn.LayerNorm(dinov2_dim),
                nn.GELU()
            )
        else:
            self.text_adapter = nn.Identity()

        # 4. Multimodal Common Space Projectors (256-D)
        self.video_common_proj = nn.Sequential(
            nn.Linear(dinov2_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim)
        )
        self.text_common_proj = nn.Sequential(
            nn.Linear(dinov2_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim)
        )

        # 5. Learnable Temperature Scale (CLIP style: init 1/0.07 ~ 14.28)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    @property
    def alpha(self) -> torch.Tensor:
        """Compatibility property for logging and checkpointing."""
        return torch.tensor(1.0)

    def encode_video(self, visual_frames: torch.Tensor, motion_frames: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_frames: [B, 3, L, 224, 224]
        Returns:
            f_video: [B, D]
            V_common: [B, common_dim] (L2 normalized)
        """
        # Accept either [B, 3, L, H, W] or [B, L, 3, H, W]
        if visual_frames.dim() == 5:
            if visual_frames.size(1) == 3:
                # [B, 3, L, H, W] -> [B, L, 3, H, W]
                visual_frames = visual_frames.permute(0, 2, 1, 3, 4)
        b, l, c, h, w = visual_frames.shape

        # Flatten frames for DINOv2: [B * L, 3, H, W]
        flat_frames = visual_frames.reshape(b * l, c, h, w)
        dino_out = self.dinov2(pixel_values=flat_frames)
        # Use [CLS] token representation
        frame_feats = dino_out.last_hidden_state[:, 0, :] # [B * L, D]
        frame_feats = frame_feats.reshape(b, l, -1) # [B, L, D]

        # Temporal attention aggregation across frames
        f_video = self.temporal_aggregator(frame_feats) # [B, D]
        V_common = F.normalize(self.video_common_proj(f_video), p=2, dim=-1)
        return f_video, V_common

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, max_len]
            attention_mask: [B, max_len]
        Returns:
            f_text: [B, D]
            T_common: [B, common_dim] (L2 normalized)
        """
        text_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (token 0)
        cls_feat = text_out.last_hidden_state[:, 0, :] # [B, 768]
        f_text = self.text_adapter(cls_feat) # [B, D]
        T_common = F.normalize(self.text_common_proj(f_text), p=2, dim=-1)
        return f_text, T_common

    def forward(
        self,
        visual_frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        motion_frames: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        f_video, V_common = self.encode_video(visual_frames, motion_frames)
        f_text, T_common = self.encode_text(input_ids, attention_mask)

        # Temperature scale clamped to max 100
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        return {
            "f_video": f_video,
            "f_text": f_text,
            "V_common": V_common,
            "T_common": T_common,
            "logit_scale": logit_scale
        }

    def compute_loss(self, T_common: torch.Tensor, V_common: torch.Tensor, logit_scale: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Symmetric Contrastive Cross-Entropy Loss (CLIP Style).
        Guarantees Anti-Collapse via mutual cross-modal negative repulsion.
        """
        b = T_common.size(0)
        labels = torch.arange(b, device=T_common.device, dtype=torch.long)

        # Scaled Cosine Similarity Matrix: [B, B]
        sim_t2v = logit_scale * torch.matmul(T_common, V_common.t())
        sim_v2t = sim_t2v.t()

        loss_t2v = F.cross_entropy(sim_t2v, labels)
        loss_v2t = F.cross_entropy(sim_v2t, labels)
        total_loss = 0.5 * (loss_t2v + loss_v2t)

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_common": total_loss.item(),
            "loss_t2v": loss_t2v.item(),
            "loss_v2t": loss_v2t.item(),
            "logit_scale": logit_scale.item()
        }
