"""
Training Script for MFGF TVPR (Thuần Tiếng Việt - 100% Vietnamese)
Huấn luyện mô hình Text-to-Video Person Retrieval (TVPR) chuyên biệt Tiếng Việt:
  - Text Encoder: PhoBERT (vinai/phobert-base-v2) + Phân đoạn từ PyVi
  - Visual Encoder: ViT (L1 khung hình ngẫu nhiên)
  - Motion Encoder: S3D (L2 khung hình liên tiếp)
  - Text Prompter: Trích xuất từ khóa ngữ nghĩa tiếng Việt và Guided Tips
  - Tối ưu hóa: Adam + Warm-up + Cosine Annealing + Dynamic Alpha Blending
"""

import argparse
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# Cấu hình UTF-8 cho console Windows để in tiếng Việt có dấu chuẩn xác
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data.dataset import build_dataloader, TVPReidDataset
from models.mfgf_main import MFGFModel
from models.dinov2_tvpr import DINOv2TVPRModel
from loss.criterion import MFGFCriterion
from utils.metrics import compute_similarity_matrix, evaluate_tvpr, print_evaluation_results
from utils.plot_benchmarks import save_training_history, load_training_history, plot_training_dashboard
from utils.paths import (
    DEFAULT_CONFIG_PATH,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    resolve_data_root,
    resolve_config_path,
    resolve_checkpoint_path,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6, base_lr: float = 1e-4):
    """Lập lịch giảm tốc độ học với Linear Warm-up và Cosine Annealing."""
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr_factor = (min_lr + (base_lr - min_lr) * cosine_factor) / base_lr
        return max(min_lr / base_lr, lr_factor)

    return LambdaLR(optimizer, lr_lambda)


def safe_load_checkpoint(path: str, map_location: Any = "cpu") -> Any:
    """Nạp file checkpoint tương thích an toàn với PyTorch 2.6+ (weights_only=False)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate(model: nn.Module, dataloader, device: torch.device, use_amp: bool = False) -> Dict[str, float]:
    """
    Đánh giá mô hình trên tập validation/test:
    Trích xuất đặc trưng câu truy vấn tiếng Việt và kho video gallery.
    """
    if len(dataloader.dataset) == 0:
        print("[!] Tập đánh giá rỗng, trả về chỉ số mặc định 0.0.")
        return {"Rank@1": 0.0, "Rank@5": 0.0, "Rank@10": 0.0, "Rank@50": 0.0, "MdR": 999.0, "mAP": 0.0, "MRR": 0.0}

    model.eval()
    text_features = []
    video_features = []
    text_labels = []
    video_labels = []
    visited_videos = set()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Đánh giá] Đang trích xuất đặc trưng", leave=False, dynamic_ncols=True):
            v_frames = batch["visual_frames"].to(device)
            m_frames = batch.get("motion_frames")
            if m_frames is not None:
                m_frames = m_frames.to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].numpy()
            vids = batch["video_id"]

            device_type = device.type
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # Mã hóa câu truy vấn tiếng Việt
                res_t = model.encode_text(input_ids, attn_mask)
                t_common = res_t[-1]
                # Mã hóa kho video gallery
                res_v = model.encode_video(v_frames, m_frames)
                v_common = res_v[-1]

            text_features.append(t_common.float().cpu())
            text_labels.extend(labels)
            v_common_cpu = v_common.float().cpu()

            for i in range(len(vids)):
                vid = vids[i]
                if vid not in visited_videos:
                    visited_videos.add(vid)
                    video_features.append(v_common_cpu[i:i+1])
                    video_labels.append(labels[i])

    if len(text_features) == 0 or len(video_features) == 0:
        return {"Rank@1": 0.0, "Rank@5": 0.0, "Rank@10": 0.0, "Rank@50": 0.0, "MdR": 999.0, "mAP": 0.0, "MRR": 0.0}

    text_features = torch.cat(text_features, dim=0) # [N_queries, 256]
    video_features = torch.cat(video_features, dim=0) # [N_gallery, 256]

    sim_matrix = compute_similarity_matrix(text_features, video_features)
    metrics = evaluate_tvpr(sim_matrix, text_labels, video_labels, ranks=[1, 5, 10, 50])
    return metrics


def train_mfgf(config_path: Optional[str] = None, resume: Optional[str] = None):
    config_path = resolve_config_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device") == "cuda" else "cpu")
    print(f"[*] Thiết bị tính toán: {device}", flush=True)

    # 1. Khởi tạo Tokenizer Tiếng Việt (PhoBERT)
    bert_name = cfg["model"].get("bert_model_name", "vinai/phobert-base-v2")
    ablation_cfg = cfg["model"].get("ablation", {})
    use_visual = bool(ablation_cfg.get("use_visual", True))
    use_motion = bool(ablation_cfg.get("use_motion", True))
    use_common = bool(ablation_cfg.get("use_common", True))
    use_d2 = bool(ablation_cfg.get("use_d2", True))

    arch = cfg["model"].get("arch", "dinov2").lower()
    is_dinov2 = (arch == "dinov2")

    print(f"[*] Chế độ: THUẦN TIẾNG VIỆT (100% Vietnamese)", flush=True)
    print(f"[*] Kiến trúc: {'DINOv2 Foundation TVPR' if is_dinov2 else 'MFGF ViT+S3D'}", flush=True)
    print(f"[*] Mô hình Text Tiếng Việt: {bert_name}", flush=True)
    if not is_dinov2:
        print(f"[*] Cấu hình Ablation: Visual(ViT)={use_visual} | Motion(S3D)={use_motion} | CommonSpace(InfoNCE)={use_common} | D2Space(Tips)={use_d2}", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
    except Exception:
        tokenizer = None
        print(f"[!] Cảnh báo: Không thể tải Tokenizer '{bert_name}', chuyển sang bộ tokenizer nội bộ.", flush=True)

    # 2. Xử lý đường dẫn dữ liệu & Khởi tạo Dataset Tiếng Việt
    data_root = resolve_data_root(cfg["data"].get("data_root", "auto"))

    dataset_name = cfg["data"].get("dataset_name", "data")
    batch_size = int(cfg["data"].get("batch_size", 12))
    num_workers = int(cfg["data"].get("num_workers", 2))
    img_size = int(cfg["model"].get("img_size", 224))
    l1 = int(cfg["model"].get("visual_frames", 4))
    l2 = int(cfg["model"].get("motion_frames", 16))
    tips_vocab_size = int(cfg["model"].get("tips_vocab_size", 1000))

    print(f"[*] Khởi tạo Dataset {dataset_name} từ {data_root}...", flush=True)
    train_dataset = TVPReidDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split="train",
        img_size=img_size,
        visual_frames=l1,
        motion_frames=l2,
        tokenizer=tokenizer,
        max_text_len=cfg["model"].get("max_text_len", 64)
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"[!] Không tìm thấy mẫu tiếng Việt nào trong tập train của {dataset_name} tại {data_root}.\n"
            f"Vui lòng kiểm tra lại file {dataset_name}.json hoặc data.json trong thư mục captions!"
        )

    all_train_captions = [item["caption"] for item in train_dataset.samples]

    if is_dinov2:
        dinov2_name = cfg["model"].get("dinov2_model_name", "facebook/dinov2-small")
        freeze_dinov2 = cfg["model"].get("freeze_dinov2", True)
        tune_dinov2_layers = int(cfg["model"].get("tune_dinov2_layers", 2))
        model = DINOv2TVPRModel(
            dinov2_name=dinov2_name,
            phobert_name=bert_name,
            visual_frames=l1,
            common_dim=cfg["model"].get("common_dim", 256),
            freeze_dinov2_backbone=freeze_dinov2,
            tune_dinov2_layers=tune_dinov2_layers
        ).to(device)
    else:
        # Xây dựng kho từ vựng tiếng Việt cho Text Prompter (MFGF legacy)
        print(f"[*] Đang xây dựng Text Prompter Keyword Bank từ {len(all_train_captions)} câu tiếng Việt...", flush=True)
        model = MFGFModel(
            img_size=img_size,
            patch_size=cfg["model"].get("patch_size", 16),
            visual_frames=l1,
            motion_frames=l2,
            embed_dim=cfg["model"].get("embed_dim", 768),
            common_dim=cfg["model"].get("common_dim", 256),
            tips_vocab_size=tips_vocab_size,
            init_alpha=cfg["model"].get("init_alpha", 0.30),
            alpha_min=cfg["model"].get("alpha_min", 0.15),
            alpha_max=cfg["model"].get("alpha_max", 0.85),
            bert_model_name=bert_name,
            pretrained_text=True,
            use_visual=use_visual,
            use_motion=use_motion,
            use_common=use_common,
            use_d2=use_d2,
            language="vi"
        ).to(device)

        # Nạp corpus vào Text Prompter để tính toán ma trận trọng số W
        if model.text_prompter is not None:
            model.text_prompter.build_vocab_from_corpus(all_train_captions, max_vocab_size=tips_vocab_size)
            model.to(device)
            train_dataset.text_prompter = model.text_prompter

    # Đảm bảo drop_last chỉ bật khi kích thước tập dữ liệu lớn hơn batch_size
    drop_last_flag = cfg["data"].get("drop_last", True) and (len(train_dataset) >= batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last_flag
    )

    # Validation DataLoader
    val_prompter = getattr(model, "text_prompter", None)
    val_loader = build_dataloader(
        data_root=data_root,
        dataset_name=dataset_name,
        split="val",
        batch_size=cfg["eval"].get("batch_size", 12),
        num_workers=num_workers,
        img_size=img_size,
        visual_frames=l1,
        motion_frames=l2,
        tokenizer=tokenizer,
        text_prompter=val_prompter,
        drop_last=False
    )

    print(f"[*] Kích thước tập Train: {len(train_dataset)} mẫu | Tập Val: {len(val_loader.dataset)} mẫu", flush=True)

    # 3. Hàm mất mát & Tối ưu hóa
    criterion = None
    if not is_dinov2:
        bce_scale = float(cfg["model"].get("bce_scale", 1.0))
        criterion = MFGFCriterion(
            temperature=cfg["model"].get("temperature", 0.05),
            distill_type="mse",
            bce_scale=bce_scale,
            use_common=use_common,
            use_d2=use_d2
        ).to(device)

    base_lr = float(cfg["train"].get("lr", 1e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    epochs = int(cfg["train"].get("epochs", 30))
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 3))
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    use_amp = bool(cfg["train"].get("use_amp", False)) and device.type == "cuda"
    device_type = device.type

    if is_dinov2:
        # Differential Learning Rates cho DINOv2 Foundation Model
        lr_dino = float(cfg["train"].get("lr_dino", 1e-5))
        lr_phobert = float(cfg["train"].get("lr_phobert", 2e-5))
        lr_head = float(cfg["train"].get("lr_head", 1e-4))
        base_lr = lr_head

        dino_params = [p for p in model.dinov2.parameters() if p.requires_grad]
        phobert_params = [p for p in model.phobert.parameters() if p.requires_grad]
        head_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("dinov2.") and not n.startswith("phobert.") and p.requires_grad
        ]

        optimizer = Adam([
            {"params": dino_params, "lr": lr_dino, "weight_decay": weight_decay},
            {"params": phobert_params, "lr": lr_phobert, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ])
        print(f"[*] Differential LR Optimizer: DINOv2={lr_dino:.1e} | PhoBERT={lr_phobert:.1e} | Head={lr_head:.1e}", flush=True)
    else:
        alpha_lr_mult = float(cfg["train"].get("alpha_lr_mult", 2.0))
        alpha_params = [model.alpha_param]
        other_params = [p for n, p in model.named_parameters() if n != "alpha_param"]

        optimizer = Adam([
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": alpha_params, "lr": base_lr * alpha_lr_mult, "weight_decay": 0.0}
        ])
    scheduler = get_lr_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, base_lr=base_lr)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    save_dir_cfg = cfg["train"].get("save_dir", "checkpoints")
    if os.path.isabs(save_dir_cfg):
        save_dir = save_dir_cfg
    else:
        save_dir = str(CHECKPOINTS_DIR) if save_dir_cfg == "checkpoints" else str(CHECKPOINTS_DIR.parent / save_dir_cfg)
    os.makedirs(save_dir, exist_ok=True)
    history_path = os.path.join(save_dir, "train_history.json")
    
    os.makedirs(str(REPORTS_DIR), exist_ok=True)
    report_img_path = str(REPORTS_DIR / "training_dashboard.png")

    history = {
        "epochs": [],
        "loss": [],
        "loss_common": [],
        "loss_d2": [],
        "alpha": [],
        "lr": [],
        "epoch_time": [],
        "eval_epochs": [],
        "rank1": [],
        "rank5": [],
        "rank10": [],
        "rank50": [],
        "mdr": [],
        "map": [],
        "mrr": []
    }

    start_epoch = 1
    best_rank1 = 0.0
    best_mdr = float("inf")

    # Xử lý tiếp tục huấn luyện (Resume Training)
    resume_target = resume if resume is not None else cfg["train"].get("resume", None)
    if resume_target:
        target_str = str(resume_target).strip().lower()
        resume_path = None

        if target_str in ["auto", "true", "latest", "latest.pth"]:
            cand = os.path.join(save_dir, "latest.pth")
            if not os.path.exists(cand):
                cand = os.path.join(save_dir, "mfgf_latest.pth")
            resume_path = cand if os.path.exists(cand) else resolve_checkpoint_path("latest.pth", preferred="latest")
        elif target_str in ["best", "best.pth"]:
            cand = os.path.join(save_dir, "best.pth")
            if not os.path.exists(cand):
                cand = os.path.join(save_dir, "mfgf_best.pth")
            resume_path = cand if os.path.exists(cand) else resolve_checkpoint_path("best.pth", preferred="best")
        else:
            cand_in_save_dir = os.path.join(save_dir, str(resume_target))
            if os.path.isfile(cand_in_save_dir):
                resume_path = cand_in_save_dir
            else:
                resume_path = resolve_checkpoint_path(str(resume_target))

        if resume_path and os.path.isfile(resume_path):
            print(f"[*] Phục hồi checkpoint huấn luyện từ: {resume_path}", flush=True)
            ckpt = safe_load_checkpoint(resume_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model_dict = model.state_dict()
            matching_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

            if len(matching_dict) < (len(model_dict) * 0.3):
                print(f"[!] Cảnh báo: Checkpoint '{resume_path}' không tương thích với kiến trúc hiện tại ({arch}).", flush=True)
                print(f"[*] Bỏ qua phục hồi trọng số không khớp, bắt đầu huấn luyện mới từ Epoch 1.", flush=True)
                start_epoch = 1
            else:
                model.load_state_dict(matching_dict, strict=False)
                saved_epoch = int(ckpt.get("epoch", 0))
                start_epoch = saved_epoch + 1
                print(f"[+] Đã nạp thành công {len(matching_dict)}/{len(model_dict)} tensor trọng số.", flush=True)

                if "best_rank1" in ckpt:
                    best_rank1 = float(ckpt["best_rank1"])
                    best_mdr = float(ckpt.get("best_mdr", float("inf")))
                elif "metrics" in ckpt and isinstance(ckpt["metrics"], dict) and "Rank@1" in ckpt["metrics"]:
                    best_rank1 = float(ckpt["metrics"]["Rank@1"])
                    best_mdr = float(ckpt.get("best_mdr", float("inf")))

            if best_rank1 == 0.0:
                best_ckpt_path = os.path.join(save_dir, "best.pth")
                if not os.path.isfile(best_ckpt_path):
                    best_ckpt_path = os.path.join(save_dir, "mfgf_best.pth")
                if os.path.isfile(best_ckpt_path):
                    try:
                        best_data = safe_load_checkpoint(best_ckpt_path, map_location="cpu")
                        if "best_rank1" in best_data:
                            best_rank1 = float(best_data["best_rank1"])
                            best_mdr = float(best_data.get("best_mdr", float("inf")))
                        elif "metrics" in best_data and isinstance(best_data["metrics"], dict) and "Rank@1" in best_data["metrics"]:
                            best_rank1 = float(best_data["metrics"]["Rank@1"])
                            best_mdr = float(best_data["metrics"].get("MdR", float("inf")))
                    except Exception:
                        pass

            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass

            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass

            if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None and use_amp:
                try:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    pass

            loaded_hist = load_training_history(history_path)
            if loaded_hist is None and "history" in ckpt and isinstance(ckpt["history"], dict):
                loaded_hist = ckpt["history"]

            if loaded_hist and isinstance(loaded_hist, dict):
                history = loaded_hist
                if "epochs" in history and history["epochs"]:
                    keep_idx = [i for i, ep in enumerate(history["epochs"]) if ep <= saved_epoch]
                    max_k = max(keep_idx) + 1 if keep_idx else 0
                    for k in ["epochs", "loss", "loss_common", "loss_d2", "alpha", "lr", "epoch_time"]:
                        if k in history and len(history[k]) >= max_k:
                            history[k] = history[k][:max_k]

                if "eval_epochs" in history and history["eval_epochs"]:
                    keep_eval_idx = [i for i, ep in enumerate(history["eval_epochs"]) if ep <= saved_epoch]
                    max_eval_k = max(keep_eval_idx) + 1 if keep_eval_idx else 0
                    for k in ["eval_epochs", "rank1", "rank5", "rank10", "rank50", "mdr", "map", "mrr"]:
                        if k in history and len(history[k]) >= max_eval_k:
                            history[k] = history[k][:max_eval_k]

            print(f"[+] Phục hồi thành công từ Epoch {saved_epoch}. Huấn luyện tiếp từ Epoch {start_epoch}.", flush=True)

    if start_epoch > epochs:
        print(f"[!] Quá trình huấn luyện đã hoàn tất {epochs} epochs.", flush=True)
        return

    print(f"[*] Bắt đầu huấn luyện từ Epoch {start_epoch} đến {epochs}...", flush=True)
    print(f"[*] Giá trị Alpha ban đầu: {model.alpha.item():.4f} | AMP: {use_amp} | Tích lũy Gradient: {grad_accum_steps}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_l_common = 0.0
        total_l_d2 = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}]", dynamic_ncols=True)
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            v_frames = batch["visual_frames"].to(device)
            m_frames = batch["motion_frames"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            f_O_tips = batch.get("f_O_tips", None)
            if f_O_tips is not None:
                f_O_tips = f_O_tips.to(device)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                if is_dinov2:
                    outputs = model(
                        visual_frames=v_frames,
                        input_ids=input_ids,
                        attention_mask=attn_mask
                    )
                    loss, loss_dict = model.compute_loss(
                        T_common=outputs["T_common"],
                        V_common=outputs["V_common"],
                        logit_scale=outputs["logit_scale"]
                    )
                else:
                    outputs = model(
                        visual_frames=v_frames,
                        motion_frames=m_frames,
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        f_O_tips=f_O_tips
                    )
                    loss, loss_dict = criterion(
                        T_common=outputs["T_common"],
                        V_common=outputs["V_common"],
                        T_D2=outputs["T_D2"],
                        V_D2=outputs["V_D2"],
                        f_tips=outputs["f_tips"],
                        alpha=outputs["alpha"]
                    )
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                grad_clip = cfg["train"].get("grad_clip_norm", 5.0)
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_l_common += loss_dict.get("loss_common", loss.item())
            total_l_d2 += loss_dict.get("loss_d2", 0.0)

            current_lr = optimizer.param_groups[0]["lr"]
            if is_dinov2:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "T2V": f"{loss_dict.get('loss_t2v', 0.0):.3f}",
                    "V2T": f"{loss_dict.get('loss_v2t', 0.0):.3f}",
                    "scale": f"{outputs['logit_scale'].item():.1f}",
                    "lr": f"{current_lr:.2e}"
                })
            else:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "L_com": f"{loss_dict['loss_common']:.3f}",
                    "L_d2": f"{loss_dict['loss_d2']:.3f}",
                    "alpha": f"{outputs['alpha'].item():.3f}",
                    "lr": f"{current_lr:.2e}"
                })

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, len(train_loader))
        avg_l_common = total_l_common / max(1, len(train_loader))
        avg_l_d2 = total_l_d2 / max(1, len(train_loader))
        status_extra = f"Scale: {outputs['logit_scale'].item():.1f}" if is_dinov2 else f"Alpha: {model.alpha.item():.4f}"
        print(f"--> Hoàn thành Epoch {epoch:02d} trong {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f} | {status_extra}", flush=True)

        history["epochs"].append(epoch)
        history["loss"].append(round(avg_loss, 4))
        history["loss_common"].append(round(avg_l_common, 4))
        history["loss_d2"].append(round(avg_l_d2, 4))
        history["alpha"].append(round(model.alpha.item(), 4))
        history["lr"].append(current_lr)
        history["epoch_time"].append(round(epoch_time, 2))

        # Đánh giá định kỳ
        if epoch % cfg["train"].get("eval_interval", 5) == 0 or epoch == epochs:
            print(f"[*] Đang thực hiện đánh giá tại Epoch {epoch}...", flush=True)
            metrics = evaluate(model, val_loader, device, use_amp=use_amp)
            print_evaluation_results(metrics, title=f"Kết quả Đánh giá @ Epoch {epoch}")

            history["eval_epochs"].append(epoch)
            history["rank1"].append(round(metrics.get("Rank@1", 0.0), 2))
            history["rank5"].append(round(metrics.get("Rank@5", 0.0), 2))
            history["rank10"].append(round(metrics.get("Rank@10", 0.0), 2))
            history["rank50"].append(round(metrics.get("Rank@50", 0.0), 2))
            history["mdr"].append(round(metrics.get("MdR", 0.0), 2))
            history["map"].append(round(metrics.get("mAP", 0.0), 2))
            history["mrr"].append(round(metrics.get("MRR", 0.0), 2))

            if metrics["Rank@1"] > best_rank1 or (metrics["Rank@1"] == best_rank1 and metrics["MdR"] < best_mdr):
                best_rank1 = metrics["Rank@1"]
                best_mdr = metrics["MdR"]
                best_path = os.path.join(save_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "arch": arch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "metrics": metrics,
                    "best_rank1": best_rank1,
                    "best_mdr": best_mdr,
                    "alpha": model.alpha.item(),
                    "config": cfg,
                    "history": history
                }, best_path)
                print(f"[+] Đã lưu checkpoint TỐT NHẤT vào {best_path} (Rank@1: {best_rank1:.2f}%, MdR: {best_mdr:.1f})", flush=True)

        latest_path = os.path.join(save_dir, "latest.pth")
        torch.save({
            "epoch": epoch,
            "arch": arch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "metrics": {"avg_loss": avg_loss},
            "best_rank1": best_rank1,
            "best_mdr": best_mdr,
            "alpha": model.alpha.item(),
            "config": cfg,
            "history": history
        }, latest_path)

        save_training_history(history, history_path)
        try:
            plot_training_dashboard(history, output_path=report_img_path)
        except Exception:
            pass

    print(f"\n[+] Quá trình huấn luyện đã hoàn tất cho {epochs} epochs!", flush=True)
    print(f"[*] Rank@1 tốt nhất đạt được: {best_rank1:.2f}% | MdR thấp nhất: {best_mdr:.1f}", flush=True)
    print(f"[*] Lịch sử huấn luyện lưu tại: {history_path}", flush=True)
    print(f"[*] Đồ thị Dashboard xuất ra tại: {report_img_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MFGF TVPR Model (Chuyên biệt Tiếng Việt)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Đường dẫn file cấu hình YAML (tự động phân giải)")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Tiếp tục huấn luyện từ checkpoint ('latest', 'best', hoặc đường dẫn cụ thể .pth)"
    )
    args = parser.parse_args()
    train_mfgf(args.config, resume=args.resume)
