"""
Main MFGF (Multielement Feature Guided Fragments Learning) TVPR Model
Combines:
  - Text Prompter (Guided Tips calculation: f_tips = W_h ⊙ f^O_tips)
  - Text Encoder (BERT [CLS] feature: f_Text)
  - Visual Encoder (ViT with additive Spatio-temporal embeddings: f_vis)
  - Motion Encoder (S3D with Separable Convs & Gating: f_mot)
  - Feature Aggregator: Concatenation + FC block -> f_ME
  - Common Space Projectors: f_Text -> T_common, f_ME -> V_common
  - FeatureConvertor: T_common -> T_D2, V_common -> V_D2
  - Learnable Alpha Blending: alpha initialized at 0.15
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_prompter import TextPrompter
from .text_encoder import TextEncoder
from .visual_encoder import VisualEncoder
from .motion_encoder import MotionEncoder
from .spaces import FeatureConvertor, CommonSpaceProjector


class FeatureAggregator(nn.Module):
    """
    Feature Aggregator for Multielement (ME) representation.
    Performs non-linear fusion on concatenated visual and motion features
    to filter out redundant spatio-temporal cues:
      f_ME = MLP([f_vis, f_mot])
    """
    def __init__(self, embed_dim: int = 768, drop_rate: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, f_vis: torch.Tensor, f_mot: torch.Tensor) -> torch.Tensor:
        concat_feat = torch.cat([f_vis, f_mot], dim=-1) # [B, 2 * embed_dim]
        f_ME = self.fusion(concat_feat) # [B, embed_dim]
        return f_ME


class MFGFModel(nn.Module):
    """
    Full MFGF Architecture for Text-to-Video Person Retrieval.
    Supports transparent on/off ablation study flags:
      - use_visual: on/off Visual Encoder (ViT)
      - use_motion: on/off Motion Encoder (S3D)
      - use_common: on/off Common Space InfoNCE contrastive
      - use_d2: on/off D^2 Guided Tips Distillation
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        visual_frames: int = 4,   # L1
        motion_frames: int = 16,  # L2
        embed_dim: int = 768,
        common_dim: int = 256,
        tips_vocab_size: int = 1000,
        init_alpha: float = 0.15,
        alpha_min: float = 0.10,
        alpha_max: float = 0.90,
        bert_model_name: str = "vinai/phobert-base-v2",
        pretrained_text: bool = True,
        use_visual: bool = True,
        use_motion: bool = True,
        use_common: bool = True,
        use_d2: bool = True,
        language: str = "vi"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.common_dim = common_dim
        self.tips_vocab_size = tips_vocab_size
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.use_visual = use_visual
        self.use_motion = use_motion
        self.use_common = use_common
        self.use_d2 = use_d2
        self.language = language

        # 1. Text Prompter (Keyword Extraction & Constant Weight W)
        if self.use_d2:
            self.text_prompter = TextPrompter(vocab_size=tips_vocab_size, language=language)
            self.text_d2_convertor = FeatureConvertor(in_dim=common_dim, out_dim=tips_vocab_size)
            self.video_d2_convertor = FeatureConvertor(in_dim=common_dim, out_dim=tips_vocab_size)
        else:
            self.text_prompter = None
            self.text_d2_convertor = None
            self.video_d2_convertor = None

        # 2. Text Encoder (BERT / PhoBERT)
        self.text_encoder = TextEncoder(
            model_name=bert_model_name,
            embed_dim=embed_dim,
            pretrained=pretrained_text
        )

        # 3. Visual Encoder (ViT with additive E_S, E_T)
        if self.use_visual:
            self.visual_encoder = VisualEncoder(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=3,
                num_frames=visual_frames,
                embed_dim=embed_dim
            )
        else:
            self.visual_encoder = None

        # 4. Motion Encoder (S3D with separable convs & Gating)
        if self.use_motion:
            self.motion_encoder = MotionEncoder(
                in_channels=3,
                num_frames=motion_frames,
                embed_dim=embed_dim
            )
        else:
            self.motion_encoder = None

        # 5. Feature Aggregator for Multielement (ME) representation
        if self.use_visual and self.use_motion:
            self.feature_aggregator = FeatureAggregator(embed_dim=embed_dim)
        elif self.use_visual:
            self.feature_aggregator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        elif self.use_motion:
            self.feature_aggregator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        else:
            raise ValueError("Ablation Error: Both 'use_visual' and 'use_motion' cannot be False simultaneously!")

        # 6. Common Space Projectors (256-D)
        self.text_common_proj = CommonSpaceProjector(in_dim=embed_dim, common_dim=common_dim)
        self.video_common_proj = CommonSpaceProjector(in_dim=embed_dim, common_dim=common_dim)

        # 7. Learnable Alpha Parameter: bounded smoothly in [alpha_min, alpha_max]
        init_clamped = min(max(init_alpha, self.alpha_min + 1e-4), self.alpha_max - 1e-4)
        norm_alpha = (init_clamped - self.alpha_min) / (self.alpha_max - self.alpha_min)
        init_logit = math.log(norm_alpha / (1.0 - norm_alpha))
        self.alpha_param = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        """Returns current dynamic alpha blending parameter bounded in [alpha_min, alpha_max]."""
        if not self.use_d2:
            return torch.tensor(1.0, device=self.alpha_param.device)
        if not self.use_common:
            return torch.tensor(0.0, device=self.alpha_param.device)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_param)

    def set_alpha(self, target_alpha: float):
        """Programmatically set alpha parameter to target value within [alpha_min, alpha_max]."""
        target_clamped = min(max(target_alpha, self.alpha_min + 1e-4), self.alpha_max - 1e-4)
        norm = (target_clamped - self.alpha_min) / (self.alpha_max - self.alpha_min)
        with torch.no_grad():
            self.alpha_param.copy_(torch.tensor(math.log(norm / (1.0 - norm)), dtype=torch.float32))

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes text into f_Text and Common Space T_common.
        Returns:
            f_Text: [B, embed_dim]
            T_common: [B, common_dim] (L2 normalized)
        """
        f_Text = self.text_encoder(input_ids, attention_mask)
        T_common = self.text_common_proj(f_Text, normalize=True)
        return f_Text, T_common

    def encode_video(
        self,
        visual_frames: Optional[torch.Tensor] = None,
        motion_frames: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Encodes video frames into f_vis, f_mot, aggregated f_ME, and Common Space V_common.
        Returns:
            f_vis: [B, embed_dim] (or zeros if ablated)
            f_mot: [B, embed_dim] (or zeros if ablated)
            f_ME: [B, embed_dim]
            V_common: [B, common_dim] (L2 normalized)
        """
        if self.use_visual and self.use_motion:
            f_vis = self.visual_encoder(visual_frames)
            f_mot = self.motion_encoder(motion_frames)
            f_ME = self.feature_aggregator(f_vis, f_mot)
        elif self.use_visual:
            f_vis = self.visual_encoder(visual_frames)
            f_mot = torch.zeros_like(f_vis)
            f_ME = self.feature_aggregator(f_vis)
        elif self.use_motion:
            f_mot = self.motion_encoder(motion_frames)
            f_vis = torch.zeros_like(f_mot)
            f_ME = self.feature_aggregator(f_mot)
        else:
            raise ValueError("Neither visual nor motion encoder is enabled!")

        V_common = self.video_common_proj(f_ME, normalize=True)
        return f_vis, f_mot, f_ME, V_common

    def forward(
        self,
        visual_frames: Optional[torch.Tensor] = None,
        motion_frames: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        f_O_tips: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Full Forward Pass (with modular ablation support):
          1. Extract f_Text and f_ME
          2. Map to Common Space: T_common, V_common
          3. Map to D^2 Space: T_D2, V_D2 (if enabled)
          4. Compute Guided Tips: f_tips (if enabled)
        """
        # 1. Text representations
        f_Text, T_common = self.encode_text(input_ids, attention_mask)

        # 2. Video representations
        f_vis, f_mot, f_ME, V_common = self.encode_video(visual_frames, motion_frames)

        # 3. Dual-Distilled Space Projections (D^2 Space)
        if self.use_d2 and self.text_d2_convertor is not None:
            T_D2 = self.text_d2_convertor(T_common) # [B, D_tips]
            V_D2 = self.video_d2_convertor(V_common) # [B, D_tips]

            # 4. Compute Guided Tips: f_tips = W_h ⊙ f^O_tips
            if f_O_tips is not None:
                f_tips = self.text_prompter(texts="", f_O_tips=f_O_tips)
            elif captions is not None:
                f_tips = self.text_prompter(texts=captions)
            else:
                device = T_common.device
                f_tips = torch.zeros(T_common.size(0), self.tips_vocab_size, device=device)
        else:
            T_D2 = None
            V_D2 = None
            f_tips = None

        return {
            "f_Text": f_Text,
            "f_vis": f_vis,
            "f_mot": f_mot,
            "f_ME": f_ME,
            "T_common": T_common,
            "V_common": V_common,
            "T_D2": T_D2,
            "V_D2": V_D2,
            "f_tips": f_tips,
            "alpha": self.alpha
        }
