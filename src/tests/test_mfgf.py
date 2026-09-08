"""
Unit and Integration Tests for MFGF TVPR Modules
Verifies tensor shapes, forward/backward passes, gating mechanisms, tips calculations, and metrics.
"""

import os
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import unittest
import numpy as np
import torch

from models.text_prompter import TextPrompter
from models.text_encoder import TextEncoder
from models.visual_encoder import VisualEncoder
from models.motion_encoder import MotionEncoder
from models.spaces import FeatureConvertor, CommonSpaceProjector
from models.mfgf_main import MFGFModel
from loss.criterion import MFGFCriterion, CommonSpaceLoss, DualDistilledLoss
from utils.metrics import compute_similarity_matrix, evaluate_tvpr


class TestMFGF(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.embed_dim = 768
        self.common_dim = 256
        self.tips_vocab_size = 50

    def test_text_prompter_and_tips(self):
        prompter = TextPrompter(vocab_size=self.tips_vocab_size)
        sample_corpus = [
            "Người phụ nữ mặc áo khoác trắng và quần xanh.",
            "Người đàn ông đi bộ với chiếc ba lô màu đen và giày đỏ."
        ]
        prompter.build_vocab_from_corpus(sample_corpus, max_vocab_size=self.tips_vocab_size)

        # Check W is registered buffer and not trainable
        self.assertIn("W", dict(prompter.named_buffers()))
        self.assertNotIn("W", dict(prompter.named_parameters()))
        self.assertAlmostEqual(prompter.W.sum().item(), 1.0, places=4)

        # Test forward Hadamard tips
        f_tips = prompter(sample_corpus)
        self.assertEqual(f_tips.shape, (2, len(prompter.W)))
        # Tips values should be non-negative
        self.assertTrue((f_tips >= 0).all())

    def test_visual_encoder_vit(self):
        vis_enc = VisualEncoder(
            img_size=224,
            patch_size=16,
            num_frames=4, # L1 = 4
            embed_dim=self.embed_dim,
            depth=2
        ).to(self.device)

        # Input [B, 3, 4, 224, 224]
        dummy_vis = torch.randn(self.batch_size, 3, 4, 224, 224, device=self.device)
        f_vis = vis_enc(dummy_vis)

        self.assertEqual(f_vis.shape, (self.batch_size, self.embed_dim))
        self.assertFalse(torch.isnan(f_vis).any())

    def test_motion_encoder_s3d(self):
        mot_enc = MotionEncoder(
            in_channels=3,
            num_frames=16, # L2 = 16
            embed_dim=self.embed_dim
        ).to(self.device)

        # Check gating parameters sigma and b exist
        self.assertIn("sigma", dict(mot_enc.named_parameters()))
        self.assertIn("b", dict(mot_enc.named_parameters()))

        # Input [B, 3, 16, 224, 224]
        dummy_mot = torch.randn(self.batch_size, 3, 16, 224, 224, device=self.device)
        f_mot = mot_enc(dummy_mot)

        self.assertEqual(f_mot.shape, (self.batch_size, self.embed_dim))
        self.assertFalse(torch.isnan(f_mot).any())

    def test_spaces_and_feature_convertor(self):
        convertor = FeatureConvertor(in_dim=self.common_dim, out_dim=self.tips_vocab_size).to(self.device)
        dummy_common = torch.randn(self.batch_size, self.common_dim, device=self.device)
        d2_out = convertor(dummy_common)

        self.assertEqual(d2_out.shape, (self.batch_size, self.tips_vocab_size))
        # Sigmoid output in (0, 1)
        self.assertTrue((d2_out >= 0).all() and (d2_out <= 1).all())

        proj = CommonSpaceProjector(in_dim=self.embed_dim, common_dim=self.common_dim).to(self.device)
        dummy_feat = torch.randn(self.batch_size, self.embed_dim, device=self.device)
        c_out = proj(dummy_feat, normalize=True)
        self.assertEqual(c_out.shape, (self.batch_size, self.common_dim))
        # L2 norm should be 1.0
        norms = torch.norm(c_out, p=2, dim=-1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=4)

    def test_full_model_forward_and_backward(self):
        model = MFGFModel(
            img_size=224,
            patch_size=16,
            visual_frames=4,
            motion_frames=16,
            embed_dim=self.embed_dim,
            common_dim=self.common_dim,
            tips_vocab_size=self.tips_vocab_size,
            init_alpha=0.15,
            pretrained_text=False # Faster for unit test
        ).to(self.device)

        criterion = MFGFCriterion(temperature=0.05).to(self.device)

        dummy_vis = torch.randn(self.batch_size, 3, 4, 224, 224, device=self.device)
        dummy_mot = torch.randn(self.batch_size, 3, 16, 224, 224, device=self.device)
        dummy_ids = torch.randint(0, 1000, (self.batch_size, 16), device=self.device)
        dummy_mask = torch.ones(self.batch_size, 16, device=self.device)
        dummy_f_O_tips = torch.randint(0, 2, (self.batch_size, self.tips_vocab_size), dtype=torch.float32, device=self.device)

        # Initial alpha
        self.assertAlmostEqual(model.alpha.item(), 0.15, places=2)

        # Forward pass
        out = model(
            visual_frames=dummy_vis,
            motion_frames=dummy_mot,
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            f_O_tips=dummy_f_O_tips
        )

        self.assertEqual(out["T_common"].shape, (self.batch_size, self.common_dim))
        self.assertEqual(out["V_common"].shape, (self.batch_size, self.common_dim))
        self.assertEqual(out["T_D2"].shape, (self.batch_size, self.tips_vocab_size))
        self.assertEqual(out["V_D2"].shape, (self.batch_size, self.tips_vocab_size))

        # Loss calculation
        loss, loss_dict = criterion(
            T_common=out["T_common"],
            V_common=out["V_common"],
            T_D2=out["T_D2"],
            V_D2=out["V_D2"],
            f_tips=out["f_tips"],
            alpha=out["alpha"]
        )

        self.assertTrue(loss.item() > 0)
        self.assertIn("loss_common", loss_dict)
        self.assertIn("loss_d2", loss_dict)

        # Backward pass
        loss.backward()

        # Check gradients exist for alpha and gating parameters
        self.assertIsNotNone(model.alpha_param.grad)
        self.assertIsNotNone(model.motion_encoder.sigma.grad)
        self.assertIsNotNone(model.motion_encoder.b.grad)

    def test_metrics_calculation(self):
        # 4 queries, 4 gallery items
        # Query labels: [0, 1, 2, 3]
        # Gallery labels: [0, 1, 2, 3]
        # Let's create an identity-like similarity matrix
        sim_matrix = np.array([
            [1.0, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.3, 0.2],
            [0.4, 0.3, 0.8, 0.1],
            [0.2, 0.9, 0.1, 0.7]  # Correct is at index 3, but index 1 is higher (0.9 > 0.7) -> rank 2
        ])
        q_labels = [0, 1, 2, 3]
        g_labels = [0, 1, 2, 3]

        metrics = evaluate_tvpr(sim_matrix, q_labels, g_labels, ranks=[1, 5])
        self.assertEqual(metrics["Rank@1"], 75.0) # 3 out of 4 correct at rank 1
        self.assertEqual(metrics["Rank@5"], 100.0)
        self.assertEqual(metrics["MdR"], 1.0) # median of [1, 1, 1, 2] is 1.0


    def test_vietnamese_text_prompter(self):
        prompter = TextPrompter(vocab_size=self.tips_vocab_size, language="vi")
        sample_corpus = [
            "Người phụ nữ mặc áo khoác màu xanh và quần đen.",
            "Người đàn ông mặc áo sơ mi trắng đang đi bộ và mang ba lô đỏ."
        ]
        prompter.build_vocab_from_corpus(sample_corpus, max_vocab_size=self.tips_vocab_size)

        # Check W buffer
        self.assertIn("W", dict(prompter.named_buffers()))
        self.assertAlmostEqual(prompter.W.sum().item(), 1.0, places=4)
        self.assertTrue(len(prompter.vocab) > 0)

        # Test forward
        f_tips = prompter(sample_corpus)
        self.assertEqual(f_tips.shape, (2, len(prompter.W)))
        self.assertTrue((f_tips >= 0).all())

    def test_ablation_study_configurations(self):
        dummy_vis = torch.randn(self.batch_size, 3, 4, 224, 224, device=self.device)
        dummy_mot = torch.randn(self.batch_size, 3, 16, 224, 224, device=self.device)
        dummy_ids = torch.randint(0, 1000, (self.batch_size, 16), device=self.device)
        dummy_mask = torch.ones(self.batch_size, 16, device=self.device)

        # 1. Ablation: Without Visual Encoder (Motion Only)
        model_no_vis = MFGFModel(
            embed_dim=self.embed_dim, common_dim=self.common_dim,
            tips_vocab_size=self.tips_vocab_size, pretrained_text=False,
            use_visual=False, use_motion=True
        ).to(self.device)
        out_no_vis = model_no_vis(motion_frames=dummy_mot, input_ids=dummy_ids, attention_mask=dummy_mask)
        self.assertEqual(out_no_vis["V_common"].shape, (self.batch_size, self.common_dim))

        # 2. Ablation: Without Motion Encoder (Visual Only)
        model_no_mot = MFGFModel(
            embed_dim=self.embed_dim, common_dim=self.common_dim,
            tips_vocab_size=self.tips_vocab_size, pretrained_text=False,
            use_visual=True, use_motion=False
        ).to(self.device)
        out_no_mot = model_no_mot(visual_frames=dummy_vis, input_ids=dummy_ids, attention_mask=dummy_mask)
        self.assertEqual(out_no_mot["V_common"].shape, (self.batch_size, self.common_dim))

        # 3. Ablation: Without D^2 Space (InfoNCE Only)
        model_no_d2 = MFGFModel(
            embed_dim=self.embed_dim, common_dim=self.common_dim,
            tips_vocab_size=self.tips_vocab_size, pretrained_text=False,
            use_d2=False
        ).to(self.device)
        crit_no_d2 = MFGFCriterion(use_d2=False).to(self.device)
        out_no_d2 = model_no_d2(visual_frames=dummy_vis, motion_frames=dummy_mot, input_ids=dummy_ids, attention_mask=dummy_mask)
        self.assertIsNone(out_no_d2["T_D2"])
        loss_no_d2, dict_no_d2 = crit_no_d2(T_common=out_no_d2["T_common"], V_common=out_no_d2["V_common"], alpha=out_no_d2["alpha"])
        self.assertEqual(dict_no_d2["loss_d2"], 0.0)

        # 4. Ablation: Without Common Space (D^2 Tips Distillation Only)
        crit_no_com = MFGFCriterion(use_common=False).to(self.device)
        crit_no_com.to(self.device)
        full_model = MFGFModel(
            embed_dim=self.embed_dim, common_dim=self.common_dim,
            tips_vocab_size=self.tips_vocab_size, pretrained_text=False
        ).to(self.device)
        out_full = full_model(visual_frames=dummy_vis, motion_frames=dummy_mot, input_ids=dummy_ids, attention_mask=dummy_mask)
        loss_no_com, dict_no_com = crit_no_com(T_D2=out_full["T_D2"], V_D2=out_full["V_D2"], f_tips=out_full["f_tips"], alpha=out_full["alpha"])
        self.assertEqual(dict_no_com["loss_common"], 0.0)
        self.assertTrue(loss_no_com.item() > 0)


if __name__ == "__main__":
    unittest.main()
