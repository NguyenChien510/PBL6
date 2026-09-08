"""
Evaluation and Inference Script for MFGF TVPR
Evaluates trained checkpoint on Test Set:
  - Cumulative Matching Characteristics (Rank@1, 5, 10, 50)
  - Median Rank (MdR) - System Stability Metric
  - Single-query inference demonstration
"""

import argparse
import os
from typing import Optional
import yaml
import torch
from transformers import AutoTokenizer

from data.dataset import build_dataloader, TVPReidDataset
from models.mfgf_main import MFGFModel
from utils.metrics import compute_similarity_matrix, evaluate_tvpr, print_evaluation_results
from utils.paths import (
    DEFAULT_CONFIG_PATH,
    BEST_CKPT_PATH,
    resolve_data_root,
    resolve_config_path,
    resolve_checkpoint_path,
)


def test_mfgf(config_path: Optional[str] = None, checkpoint_path: Optional[str] = None, query_text: str = None):
    config_path = resolve_config_path(config_path)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path, preferred="best")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device") == "cuda" else "cpu")
    print(f"[*] Evaluation Device: {device}")

    data_root = resolve_data_root(cfg["data"].get("data_root", "auto"))
    dataset_name = cfg["data"].get("dataset_name", "data")
    img_size = cfg["model"]["img_size"]
    l1 = cfg["model"]["visual_frames"]
    l2 = cfg["model"]["motion_frames"]
    tips_vocab_size = cfg["model"]["tips_vocab_size"]
    bert_name = cfg["model"].get("bert_model_name", "vinai/phobert-base-v2")
    language = cfg["model"].get("language", "vi")
    ablation_cfg = cfg["model"].get("ablation", {})

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
    except Exception:
        tokenizer = None

    # Instantiate Model
    model = MFGFModel(
        img_size=img_size,
        patch_size=cfg["model"]["patch_size"],
        visual_frames=l1,
        motion_frames=l2,
        embed_dim=cfg["model"]["embed_dim"],
        common_dim=cfg["model"]["common_dim"],
        tips_vocab_size=tips_vocab_size,
        init_alpha=cfg["model"]["init_alpha"],
        alpha_min=cfg["model"].get("alpha_min", 0.15),
        alpha_max=cfg["model"].get("alpha_max", 0.85),
        bert_model_name=bert_name,
        pretrained_text=False, # weights loaded from checkpoint
        use_visual=bool(ablation_cfg.get("use_visual", True)),
        use_motion=bool(ablation_cfg.get("use_motion", True)),
        use_common=bool(ablation_cfg.get("use_common", True)),
        use_d2=bool(ablation_cfg.get("use_d2", True)),
        language=language
    ).to(device)

    # Load Weights
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[+] Successfully loaded checkpoint from {checkpoint_path}")
        if "alpha" in ckpt:
            print(f"[*] Trained Alpha Value: {ckpt['alpha']:.4f}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.eval()

    # Load Test DataLoader
    test_loader = build_dataloader(
        data_root=data_root,
        dataset_name=dataset_name,
        split="test",
        batch_size=cfg["eval"].get("batch_size", 16),
        num_workers=cfg["data"].get("num_workers", 2),
        img_size=img_size,
        visual_frames=l1,
        motion_frames=l2,
        tokenizer=tokenizer,
        text_prompter=None,
        language=language,
        drop_last=False
    )
    print(f"[*] Loaded Test Dataset: {len(test_loader.dataset)} query-video pairs.")

    # 1. Custom Text Query Retrieval Mode
    if query_text:
        print(f"\n[*] Processing Custom Text Query: \"{query_text}\"")
        token_query = query_text
        if language == "vi":
            try:
                from pyvi import ViTokenizer
                token_query = ViTokenizer.tokenize(query_text)
            except Exception:
                pass

        if tokenizer is not None:
            encoding = tokenizer(
                token_query,
                padding="max_length",
                truncation=True,
                max_length=cfg["model"]["max_text_len"],
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attn_mask = encoding["attention_mask"].to(device)
        else:
            input_ids = torch.zeros(1, cfg["model"]["max_text_len"], dtype=torch.long, device=device)
            attn_mask = torch.ones(1, cfg["model"]["max_text_len"], dtype=torch.long, device=device)

        with torch.no_grad():
            _, q_common = model.encode_text(input_ids, attn_mask) # [1, 256]

        # Extract Gallery
        gallery_vids = []
        gallery_features = []
        visited = set()

        with torch.no_grad():
            for batch in test_loader:
                v_frames = batch["visual_frames"].to(device)
                m_frames = batch["motion_frames"].to(device)
                vids = batch["video_id"]
                _, _, _, v_common = model.encode_video(v_frames, m_frames)
                for i in range(len(vids)):
                    if vids[i] not in visited:
                        visited.add(vids[i])
                        gallery_vids.append(vids[i])
                        gallery_features.append(v_common[i:i+1].cpu())

        gallery_features = torch.cat(gallery_features, dim=0)
        sim_scores = torch.matmul(q_common.cpu(), gallery_features.t()).squeeze(0).numpy()
        ranked_indices = (-sim_scores).argsort()

        print("\n=== Top 5 Retrieved Persons ===")
        for rank, idx in enumerate(ranked_indices[:5], 1):
            print(f"Rank {rank}: {gallery_vids[idx]} (Similarity Score: {sim_scores[idx]:.4f})")
        return

    # 2. Benchmark Evaluation on Full Test Set
    text_features = []
    video_features = []
    text_labels = []
    video_labels = []
    visited_videos = set()

    print("[*] Extracting multimodal features on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            v_frames = batch["visual_frames"].to(device)
            m_frames = batch["motion_frames"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].numpy()
            vids = batch["video_id"]

            # Query text features
            _, t_common = model.encode_text(input_ids, attn_mask)
            text_features.append(t_common.cpu())
            text_labels.extend(labels)

            # Video gallery features
            _, _, _, v_common = model.encode_video(v_frames, m_frames)
            v_common_cpu = v_common.cpu()

            for i in range(len(vids)):
                vid = vids[i]
                if vid not in visited_videos:
                    visited_videos.add(vid)
                    video_features.append(v_common_cpu[i:i+1])
                    video_labels.append(labels[i])

    text_features = torch.cat(text_features, dim=0)
    video_features = torch.cat(video_features, dim=0)

    print(f"[*] Total Queries: {len(text_labels)} | Unique Gallery Videos: {len(video_labels)}")
    sim_matrix = compute_similarity_matrix(text_features, video_features)
    metrics = evaluate_tvpr(sim_matrix, text_labels, video_labels, ranks=[1, 5, 10, 50])

    print_evaluation_results(metrics, title=f"MFGF TVPR Final Test Benchmark ({dataset_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MFGF TVPR Model")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML (auto-resolved)")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_CKPT_PATH), help="Path to checkpoint (auto-resolved)")
    parser.add_argument("--query", type=str, default=None, help="Optional text query string for inference demo")
    args = parser.parse_args()

    test_mfgf(args.config, args.checkpoint, args.query)
