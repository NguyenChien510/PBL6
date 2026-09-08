"""
Loss Functions for MFGF TVPR
Implements:
  1. L_common: Bidirectional Contrastive Softmax loss with temperature theta=0.05
  2. L_D2: Dual-Distilled loss with batch size multiplier B:
       L_D2 = L(T^D2, V^D2) + B * [BCE(Tips, T^D2) + BCE(Tips, V^D2)]
  3. L_MFGF: Total blended objective:
       L_MFGF = alpha * L_common + (1 - alpha) * L_D2
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CommonSpaceLoss(nn.Module):
    """
    Bidirectional Contrastive Softmax Loss (InfoNCE) in the 256-D Common Space.
    Temperature theta = 0.05.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, T_common: torch.Tensor, V_common: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T_common: Normalized text embeddings [B, common_dim]
            V_common: Normalized video embeddings [B, common_dim]
        Returns:
            L_common: Scalar loss
        """
        b = T_common.size(0)
        labels = torch.arange(b, device=T_common.device, dtype=torch.long)

        # Compute cosine similarity matrix (embeddings are L2 normalized)
        sim_matrix = torch.matmul(T_common, V_common.t()) / self.temperature # [B, B]

        # Text-to-Video and Video-to-Text Cross Entropy
        loss_t2v = F.cross_entropy(sim_matrix, labels)
        loss_v2t = F.cross_entropy(sim_matrix.t(), labels)

        return 0.5 * (loss_t2v + loss_v2t)


class DualDistilledLoss(nn.Module):
    """
    Dual-Distilled Loss in D^2 Space:
      L_D2 = L(T^D2, V^D2) + bce_scale * [BCE(Tips, T^D2) + BCE(Tips, V^D2)]
    Balanced scaling ensures L_D2 doesn't overpower L_common gradients.
    """
    def __init__(self, distill_type: str = "mse", bce_scale: float = 1.0):
        super().__init__()
        self.distill_type = distill_type
        self.bce_scale = bce_scale
        self.bce_loss = nn.BCELoss(reduction="mean")

    def forward(self, T_D2: torch.Tensor, V_D2: torch.Tensor, Tips: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            T_D2: Text D^2 space prediction in (0, 1) [B, D_tips]
            V_D2: Video D^2 space prediction in (0, 1) [B, D_tips]
            Tips: Target Guided Tips representation [B, D_tips]
        Returns:
            L_D2: Total D^2 loss
            metrics: breakdown dictionary
        """
        # 1. Distillation loss between cross-modal distributions
        if self.distill_type == "mse":
            l_distill = F.mse_loss(T_D2, V_D2)
        else:
            # Symmetric KL divergence
            eps = 1e-8
            p = torch.clamp(T_D2, eps, 1.0 - eps)
            q = torch.clamp(V_D2, eps, 1.0 - eps)
            l_distill = 0.5 * (
                F.kl_div(p.log(), q, reduction="batchmean") +
                F.kl_div(q.log(), p, reduction="batchmean")
            )

        # Ensure Tips targets are clamped to [0, 1] for BCE and on the same device
        target_tips = torch.clamp(Tips, 0.0, 1.0).to(T_D2.device).float()
        # Avoid zero/one log issues
        t_d2_clamped = torch.clamp(T_D2.float(), 1e-7, 1.0 - 1e-7)
        v_d2_clamped = torch.clamp(V_D2.float(), 1e-7, 1.0 - 1e-7)

        # 2. Guidance BCE Losses (run in float32 without autocast for numerical stability)
        device_type = T_D2.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            bce_t = F.binary_cross_entropy(t_d2_clamped, target_tips)
            bce_v = F.binary_cross_entropy(v_d2_clamped, target_tips)

        # Formula: L_D2 = L(T^D2, V^D2) + bce_scale * [BCE(Tips, T^D2) + BCE(Tips, V^D2)]
        l_d2 = l_distill + self.bce_scale * (bce_t + bce_v)

        breakdown = {
            "l_distill": l_distill.item(),
            "bce_t": bce_t.item(),
            "bce_v": bce_v.item(),
            "l_d2": l_d2.item()
        }
        return l_d2, breakdown


class MFGFCriterion(nn.Module):
    """
    Combined MFGF Criterion:
      L_MFGF = alpha * L_common + (1 - alpha) * L_D2
    Supports transparent on/off ablation study flags:
      - use_common: on/off InfoNCE Contrastive Loss
      - use_d2: on/off Dual-Distilled Tips Guidance Loss
    """
    def __init__(
        self,
        temperature: float = 0.05,
        distill_type: str = "mse",
        bce_scale: float = 1.0,
        use_common: bool = True,
        use_d2: bool = True
    ):
        super().__init__()
        self.use_common = use_common
        self.use_d2 = use_d2
        self.common_loss = CommonSpaceLoss(temperature=temperature) if use_common else None
        self.d2_loss = DualDistilledLoss(distill_type=distill_type, bce_scale=bce_scale) if use_d2 else None

    def forward(
        self,
        T_common: Optional[torch.Tensor] = None,
        V_common: Optional[torch.Tensor] = None,
        T_D2: Optional[torch.Tensor] = None,
        V_D2: Optional[torch.Tensor] = None,
        f_tips: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete blended MFGF loss (or ablated subset).
        """
        device = alpha.device if alpha is not None else (T_common.device if T_common is not None else torch.device("cpu"))
        l_common = torch.tensor(0.0, device=device)
        l_d2 = torch.tensor(0.0, device=device)
        d2_breakdown = {}

        if self.use_common and self.common_loss is not None:
            if T_common is not None and V_common is not None:
                l_common = self.common_loss(T_common, V_common)

        if self.use_d2 and self.d2_loss is not None:
            if T_D2 is not None and V_D2 is not None and f_tips is not None:
                l_d2, d2_breakdown = self.d2_loss(T_D2, V_D2, f_tips)

        # Dynamic Alpha Blending according to active ablation components
        if self.use_common and self.use_d2:
            alpha_val = alpha if alpha is not None else torch.tensor(0.5, device=device)
            l_mfgf = alpha_val * l_common + (1.0 - alpha_val) * l_d2
        elif self.use_common:
            l_mfgf = l_common
        elif self.use_d2:
            l_mfgf = l_d2
        else:
            raise ValueError("Ablation Error: Both 'use_common' and 'use_d2' cannot be turned off simultaneously!")

        curr_alpha = alpha.item() if alpha is not None else 1.0
        loss_dict = {
            "loss_total": l_mfgf.item(),
            "loss_common": l_common.item(),
            "loss_d2": l_d2.item(),
            "alpha": curr_alpha,
            **d2_breakdown
        }

        return l_mfgf, loss_dict
