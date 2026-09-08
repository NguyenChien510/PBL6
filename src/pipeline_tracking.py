"""
Pipeline Tracking & Text-to-Video Matching for MFGF TVPR
Integrates:
  1. Robust Continuous Person Detection & Tracking (SORT/ByteTrack style with lifecycle)
  2. Gap interpolation & Velocity motion prediction
  3. Out-of-Camera exit detection (stops when target leaves camera view)
  4. MFGF Text-to-Video Feature Extraction & Cross-modal Cosine Matching
  5. Rendering Output Video with High-Visibility Neon Tracking Box & Dynamic Status Banner
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
import yaml

from transformers import AutoTokenizer
from models.mfgf_main import MFGFModel
from models.dinov2_tvpr import DINOv2TVPRModel
from data.transforms import VideoTransform, sample_video_frames
from utils.paths import (
    DEFAULT_CONFIG_PATH,
    LATEST_CKPT_PATH,
    BEST_CKPT_PATH,
    OUTPUTS_DIR,
    resolve_config_path,
    resolve_checkpoint_path,
    resolve_output_path,
)


class RobustPersonTracker:
    """
    Robust Multi-Person Tracker with persistent identity lifecycle.
    Keeps tracks alive through occlusions and detector flicker (up to max_lost frames),
    interpolates gaps smoothly, and detects when a person exits the camera view.
    """
    def __init__(self, device: torch.device, max_lost_frames: int = 30):
        self.device = device
        self.max_lost_frames = max_lost_frames
        self.backend = "opencv"
        self.model = None

        # 1. Check if ultralytics YOLO is available
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.backend = "yolo"
            print("[+] Using YOLOv8 Tracker backend.", flush=True)
        except Exception:
            # 2. Check if torchvision Faster R-CNN MobileNet is available
            try:
                import torchvision.models.detection as tv_det
                weights = tv_det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.model = tv_det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()
                self.backend = "torchvision"
                print("[+] Using Torchvision MobileNet-FPN Person Detector backend.", flush=True)
            except Exception:
                # 3. Fallback to OpenCV HOG
                self.model = cv2.HOGDescriptor()
                self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.backend = "opencv"
                print("[+] Using OpenCV HOG Person Detector backend.", flush=True)

    def _detect_frame(self, frame: np.ndarray, width: int, height: int, conf_thresh: float = 0.4) -> List[List[int]]:
        """Detects people in a single frame and returns list of [x1, y1, x2, y2]."""
        detections = []

        if self.backend == "yolo":
            results = self.model(frame, classes=[0], conf=conf_thresh, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    coords = b.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, coords[0]), max(0, coords[1])
                    x2, y2 = min(width, coords[2]), min(height, coords[3])
                    if x2 - x1 > 15 and y2 - y1 > 35:
                        detections.append([x1, y1, x2, y2])

        elif self.backend == "torchvision":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device).unsqueeze(0)
            with torch.no_grad():
                preds = self.model(tensor)[0]
            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            boxes = preds["boxes"].cpu().numpy().astype(int)

            for idx, label in enumerate(labels):
                if label == 1 and scores[idx] >= conf_thresh:  # 1 = person
                    b = boxes[idx]
                    x1, y1 = max(0, b[0]), max(0, b[1])
                    x2, y2 = min(width, b[2]), min(height, b[3])
                    if x2 - x1 > 15 and y2 - y1 > 35:
                        detections.append([x1, y1, x2, y2])
        else:
            # OpenCV HOG
            boxes, weights_scores = self.model.detectMultiScale(
                frame, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            for (x, y, w, h) in boxes:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                if x2 - x1 > 15 and y2 - y1 > 35:
                    detections.append([x1, y1, x2, y2])

        return detections

    def track_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, Dict[int, List[int]]], Dict[int, Dict[str, Any]], int, int, float, int]:
        """
        Processes video frames, detects people and maintains persistent continuous tracklets.
        Returns:
            tracklets_crops: {track_id: [cropped_person_bgr_frames]}
            tracklets_boxes: {track_id: {frame_idx: [x1, y1, x2, y2]}}
            tracklets_meta: {track_id: {'start_frame': int, 'end_frame': int, 'out_of_camera': bool, 'exit_frame': Optional[int]}}
            width, height, fps, total_processed_frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_video_frames <= 0:
            total_video_frames = 100

        total_to_process = total_video_frames if max_frames is None else min(total_video_frames, max_frames)

        tracks: Dict[int, Dict[str, Any]] = {}
        next_tid = 1
        frame_idx = 0

        # Read frames sequentially
        while cap.isOpened() and frame_idx < total_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self._detect_frame(frame, width, height)

            # Match detections to active tracks
            matched_dets = set()
            matched_tracks = set()

            # Active tracks: tracks that were updated recently
            active_tids = [tid for tid, t in tracks.items() if t["lost"] <= self.max_lost_frames and not t.get("out_of_camera", False)]

            for tid in active_tids:
                t = tracks[tid]
                last_box = t["boxes"][t["last_frame"]]

                # Velocity predicted box
                pred_box = [
                    int(last_box[0] + t["vx"]),
                    int(last_box[1] + t["vy"]),
                    int(last_box[2] + t["vx"]),
                    int(last_box[3] + t["vy"]),
                ]

                best_score = 0.0
                best_d_idx = -1

                for d_idx, det in enumerate(detections):
                    if d_idx in matched_dets:
                        continue

                    # IoU
                    ix1, iy1 = max(pred_box[0], det[0]), max(pred_box[1], det[1])
                    ix2, iy2 = min(pred_box[2], det[2]), min(pred_box[3], det[3])
                    iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    uarea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) + (det[2] - det[0]) * (det[3] - det[1]) - iarea
                    iou = iarea / max(uarea, 1e-6)

                    # Normalized center distance
                    cx1, cy1 = (pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0
                    cx2, cy2 = (det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0
                    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                    diag = max(np.sqrt((det[2] - det[0]) ** 2 + (det[3] - det[1]) ** 2), 10.0)

                    # Match if overlapping or close in proximity
                    if iou > 0.15 or dist < diag * 0.85:
                        score = iou + max(0.0, 1.0 - dist / diag)
                        if score > best_score:
                            best_score = score
                            best_d_idx = d_idx

                if best_d_idx >= 0:
                    det = detections[best_d_idx]
                    matched_dets.add(best_d_idx)
                    matched_tracks.add(tid)

                    # Interpolate any missing frames smoothly
                    gap = frame_idx - t["last_frame"]
                    if gap > 1:
                        for gf in range(t["last_frame"] + 1, frame_idx):
                            alpha = (gf - t["last_frame"]) / float(gap)
                            inter_box = [int(last_box[k] + alpha * (det[k] - last_box[k])) for k in range(4)]
                            t["boxes"][gf] = inter_box

                    # Update velocity moving average
                    cur_vx = (det[0] - last_box[0] + det[2] - last_box[2]) / (2.0 * max(gap, 1))
                    cur_vy = (det[1] - last_box[1] + det[3] - last_box[3]) / (2.0 * max(gap, 1))
                    t["vx"] = 0.6 * cur_vx + 0.4 * t["vx"]
                    t["vy"] = 0.6 * cur_vy + 0.4 * t["vy"]

                    t["boxes"][frame_idx] = det
                    t["last_frame"] = frame_idx
                    t["lost"] = 0

                    # Save crop for feature encoding
                    x1, y1, x2, y2 = det
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        t["crops"].append(crop)
                else:
                    t["lost"] += 1
                    # Check if track moved near camera border and exited
                    lbox = t["boxes"][t["last_frame"]]
                    at_border = (lbox[0] <= 8 or lbox[2] >= width - 8 or lbox[1] <= 8 or lbox[3] >= height - 8)
                    if at_border and t["lost"] >= 10:
                        t["out_of_camera"] = True
                        t["exit_frame"] = t["last_frame"]

            # Initialize new tracks for unmatched detections
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_dets:
                    x1, y1, x2, y2 = det
                    crop = frame[y1:y2, x1:x2]
                    tracks[next_tid] = {
                        "boxes": {frame_idx: det},
                        "crops": [crop] if crop.size > 0 else [],
                        "start_frame": frame_idx,
                        "last_frame": frame_idx,
                        "lost": 0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "out_of_camera": False,
                        "exit_frame": None,
                    }
                    next_tid += 1

            frame_idx += 1
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(
                    (frame_idx / total_to_process) * 0.45,
                    f"Đang phân tích & theo vết người... {frame_idx}/{total_to_process} frames"
                )

        cap.release()

        # Check final out_of_camera status for tracks
        for tid, t in tracks.items():
            if not t.get("out_of_camera", False):
                lbox = t["boxes"][t["last_frame"]]
                at_border = (lbox[0] <= 10 or lbox[2] >= width - 10 or lbox[1] <= 10 or lbox[3] >= height - 10)
                if at_border and t["last_frame"] < (total_to_process - 5):
                    t["out_of_camera"] = True
                    t["exit_frame"] = t["last_frame"]
                elif t["last_frame"] >= (total_to_process - 5):
                    t["out_of_camera"] = False  # Stayed until the end of video!
                    t["exit_frame"] = total_to_process - 1

        # Tracklet Stitching: merge broken sequential fragments of the same person
        tracks = self._stitch_tracks(tracks, width, height, fps)

        # Build return dictionaries
        tracklets_crops = {tid: t["crops"] for tid, t in tracks.items() if len(t["boxes"]) >= 3}
        tracklets_boxes = {tid: t["boxes"] for tid, t in tracks.items() if len(t["boxes"]) >= 3}
        tracklets_meta = {
            tid: {
                "start_frame": min(t["boxes"].keys()),
                "end_frame": max(t["boxes"].keys()),
                "out_of_camera": t.get("out_of_camera", False),
                "exit_frame": t.get("exit_frame", max(t["boxes"].keys())),
                "total_frames": len(t["boxes"])
            }
            for tid, t in tracks.items() if len(t["boxes"]) >= 3
        }

        # Fallback: if no person was detected, treat full frame as track 1
        if len(tracklets_boxes) == 0:
            cap = cv2.VideoCapture(video_path)
            crops = []
            f_idx = 0
            while cap.isOpened() and f_idx < total_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                crops.append(frame)
                f_idx += 1
            cap.release()
            tracklets_crops[1] = crops
            tracklets_boxes[1] = {i: [0, 0, width, height] for i in range(len(crops))}
            tracklets_meta[1] = {
                "start_frame": 0,
                "end_frame": len(crops) - 1,
                "out_of_camera": False,
                "exit_frame": len(crops) - 1,
                "total_frames": len(crops)
            }

        return tracklets_crops, tracklets_boxes, tracklets_meta, width, height, fps, total_to_process

    def _stitch_tracks(self, tracks: Dict[int, Dict[str, Any]], width: int, height: int, fps: float) -> Dict[int, Dict[str, Any]]:
        """Stitches fragmented tracklets belonging to the same person."""
        sorted_tids = sorted(tracks.keys(), key=lambda tid: min(tracks[tid]["boxes"].keys()))
        merged_into: Dict[int, int] = {}

        for i in range(len(sorted_tids)):
            tid_a = sorted_tids[i]
            while tid_a in merged_into:
                tid_a = merged_into[tid_a]
            track_a = tracks[tid_a]
            end_a = max(track_a["boxes"].keys())
            box_a = track_a["boxes"][end_a]

            for j in range(i + 1, len(sorted_tids)):
                tid_b = sorted_tids[j]
                while tid_b in merged_into:
                    tid_b = merged_into[tid_b]
                if tid_a == tid_b:
                    continue

                track_b = tracks[tid_b]
                start_b = min(track_b["boxes"].keys())

                gap = start_b - end_a
                # If track B starts shortly after track A ends (within 2 seconds)
                if 0 < gap <= int(fps * 2.0):
                    box_b = track_b["boxes"][start_b]
                    ca_x, ca_y = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
                    cb_x, cb_y = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
                    dist = np.sqrt((ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2)
                    diag = max(np.sqrt((box_a[2] - box_a[0]) ** 2 + (box_a[3] - box_a[1]) ** 2), 20.0)

                    # If spatial distance is consistent with human walking speed
                    if dist <= diag * 1.8:
                        # Merge B into A
                        # Interpolate gap
                        for gf in range(end_a + 1, start_b):
                            alpha = (gf - end_a) / float(gap)
                            inter = [int(box_a[k] + alpha * (box_b[k] - box_a[k])) for k in range(4)]
                            track_a["boxes"][gf] = inter

                        for bf, b_box in track_b["boxes"].items():
                            track_a["boxes"][bf] = b_box

                        track_a["crops"].extend(track_b["crops"])
                        track_a["last_frame"] = max(track_b["boxes"].keys())
                        track_a["out_of_camera"] = track_b.get("out_of_camera", False)
                        track_a["exit_frame"] = track_b.get("exit_frame", None)

                        merged_into[tid_b] = tid_a
                        end_a = track_a["last_frame"]
                        box_a = track_a["boxes"][end_a]

        cleaned_tracks = {tid: t for tid, t in tracks.items() if tid not in merged_into}
        return cleaned_tracks


class MFGFTrackingEngine:
    """
    Loads MFGF model checkpoint, matches query text with continuous video tracks,
    and renders output video with seamless persistent tracking.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        config_path = resolve_config_path(config_path)
        checkpoint_path = resolve_checkpoint_path(checkpoint_path, preferred="latest")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = self.cfg["model"].get("img_size", 224)
        self.l1 = self.cfg["model"].get("visual_frames", 4)
        self.l2 = self.cfg["model"].get("motion_frames", 16)
        self.transform = VideoTransform(img_size=self.img_size, is_train=False)

        bert_name = self.cfg["model"].get("bert_model_name", "vinai/phobert-base-v2")
        self.language = self.cfg["model"].get("language", "vi")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        except Exception:
            self.tokenizer = None

        ablation_cfg = self.cfg["model"].get("ablation", {})
        arch = self.cfg["model"].get("arch", "dinov2").lower()

        # Kiểm tra xem checkpoint có chỉ định kiến trúc hay không
        ckpt = None
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "arch" in ckpt:
                arch = str(ckpt["arch"]).lower()
            elif isinstance(ckpt, dict) and "arch" not in self.cfg.get("model", {}):
                state_dict_keys = (ckpt.get("model_state_dict") or ckpt).keys()
                if any(k.startswith("dinov2.") for k in state_dict_keys):
                    arch = "dinov2"
                elif any(k.startswith("visual_encoder.") or k.startswith("motion_encoder.") for k in state_dict_keys):
                    arch = "mfgf"

        self.arch = arch
        print(f"[*] Pipeline Tracking sử dụng kiến trúc: {'DINOv2 Foundation' if arch == 'dinov2' else 'MFGF ViT+S3D'}", flush=True)

        if arch == "dinov2":
            dinov2_name = self.cfg["model"].get("dinov2_model_name", "facebook/dinov2-small")
            self.model = DINOv2TVPRModel(
                dinov2_name=dinov2_name,
                phobert_name=bert_name,
                visual_frames=self.l1,
                common_dim=self.cfg["model"].get("common_dim", 256),
                freeze_dinov2_backbone=True,
                tune_dinov2_layers=0
            ).to(self.device).eval()
        else:
            self.model = MFGFModel(
                img_size=self.img_size,
                patch_size=self.cfg["model"].get("patch_size", 16),
                visual_frames=self.l1,
                motion_frames=self.l2,
                embed_dim=self.cfg["model"].get("embed_dim", 768),
                common_dim=self.cfg["model"].get("common_dim", 256),
                tips_vocab_size=self.cfg["model"].get("tips_vocab_size", 1000),
                init_alpha=self.cfg["model"].get("init_alpha", 0.30),
                alpha_min=self.cfg["model"].get("alpha_min", 0.15),
                alpha_max=self.cfg["model"].get("alpha_max", 0.85),
                bert_model_name=bert_name,
                pretrained_text=False,
                use_visual=bool(ablation_cfg.get("use_visual", True)),
                use_motion=bool(ablation_cfg.get("use_motion", True)),
                use_common=bool(ablation_cfg.get("use_common", True)),
                use_d2=bool(ablation_cfg.get("use_d2", True)),
                language=self.language
            ).to(self.device).eval()

        if ckpt is not None:
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model_dict = self.model.state_dict()
            filtered_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        skipped_keys.append(k)
            self.model.load_state_dict(filtered_dict, strict=False)
            if skipped_keys:
                print(f"[*] Đã nạp checkpoint từ {checkpoint_path} (bỏ qua {len(skipped_keys)} tham số lệch kích thước)", flush=True)
            else:
                print(f"[+] Loaded model checkpoint from {checkpoint_path} ({len(filtered_dict)}/{len(model_dict)} tensors)", flush=True)
        else:
            print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.", flush=True)

        self.tracker = RobustPersonTracker(self.device, max_lost_frames=30)

    def _prepare_tracklet_tensor(self, frames_bgr: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        total_f = len(frames_bgr)
        l1_idx, l2_idx = sample_video_frames(total_frames=total_f, l1_random=self.l1, l2_continuous=self.l2, is_train=False)

        tensor_frames = []
        for img_bgr in frames_bgr:
            if img_bgr is None or img_bgr.size == 0:
                img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size))
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t_frame = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor_frames.append(t_frame)

        fallback = tensor_frames[0] if len(tensor_frames) > 0 else torch.zeros(3, self.img_size, self.img_size)
        l1_t = torch.stack([tensor_frames[i] if i < total_f else fallback for i in l1_idx], dim=0)
        l2_t = torch.stack([tensor_frames[i] if i < total_f else fallback for i in l2_idx], dim=0)

        v_frames = self.transform(l1_t).unsqueeze(0).to(self.device)
        m_frames = self.transform(l2_t).unsqueeze(0).to(self.device)
        return v_frames, m_frames

    @staticmethod
    def _smooth_boxes(boxes_dict: Dict[int, List[int]], alpha: float = 0.75) -> Dict[int, List[int]]:
        """Applies exponential moving average to eliminate bounding box jitter."""
        sorted_keys = sorted(boxes_dict.keys())
        smoothed = {}
        prev = None
        for f in sorted_keys:
            cur = boxes_dict[f]
            if prev is None:
                smoothed[f] = cur
                prev = [float(x) for x in cur]
            else:
                s = [int(alpha * cur[i] + (1.0 - alpha) * prev[i]) for i in range(4)]
                smoothed[f] = s
                prev = [float(x) for x in s]
        return smoothed

    def process_and_track(
        self,
        video_path: str,
        query_text: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        top_k: int = 1
    ) -> Dict[str, Any]:
        """
        Runs the complete robust tracking pipeline:
        Detect/Track -> MFGF Cross-modal Matching -> Smoothing -> Render Full Video with Top-K Targets.
        """
        output_path = resolve_output_path(output_path, default_filename="result_tracked.mp4")

        if progress_callback:
            progress_callback(0.05, "Đang quét và theo vết người liên tục trong video...")

        # 1. Track Video Continuously
        tracklets_crops, tracklets_boxes, tracklets_meta, width, height, fps, total_frames = self.tracker.track_video(
            video_path, max_frames=max_frames, progress_callback=progress_callback
        )

        if len(tracklets_boxes) == 0:
            raise RuntimeError("Không tìm thấy người nào trong video để theo vết!")

        if progress_callback:
            progress_callback(0.48, f"Tìm thấy {len(tracklets_boxes)} quỹ đạo người. Đang đối soát ngữ nghĩa văn bản...")

        # 2. Encode Query Text (Vietnamese word segmentation if language is Vietnamese)
        token_query = query_text
        if self.language == "vi":
            try:
                from pyvi import ViTokenizer
                token_query = ViTokenizer.tokenize(query_text)
            except Exception:
                pass

        if self.tokenizer is not None:
            enc = self.tokenizer(
                token_query,
                padding="max_length",
                truncation=True,
                max_length=self.cfg["model"].get("max_text_len", 64),
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)
        else:
            input_ids = torch.zeros(1, 64, dtype=torch.long, device=self.device)
            attn_mask = torch.ones(1, 64, dtype=torch.long, device=self.device)

        with torch.no_grad():
            res_q = self.model.encode_text(input_ids, attn_mask)
            q_common = res_q[-1]
            q_norm = F.normalize(q_common, p=2, dim=-1)

        # 3. Encode Tracklets & Compute Similarity
        track_scores: Dict[int, float] = {}
        for idx, (tid, crops) in enumerate(tracklets_crops.items()):
            if len(crops) < 4:
                crops = crops * 4 if len(crops) > 0 else [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)] * 4

            v_frames, m_frames = self._prepare_tracklet_tensor(crops)
            with torch.no_grad():
                res_v = self.model.encode_video(v_frames, m_frames)
                v_common = res_v[-1]
                v_norm = F.normalize(v_common, p=2, dim=-1)
                sim = torch.matmul(q_norm, v_norm.t()).item()
                track_scores[tid] = float(sim)

            if progress_callback:
                progress_callback(0.48 + (idx + 1) / len(tracklets_crops) * 0.22, f"Đang đối soát người #{tid}...")

        # 4. Determine Top-K Target Persons
        sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        actual_k = max(1, min(int(top_k), len(sorted_tracks)))
        selected_tracks = sorted_tracks[:actual_k]

        # Distinct BGR color palette for Top-K ranks
        RANK_COLORS = [
            (0, 255, 127),   # Rank 1: Neon Green
            (255, 200, 0),   # Rank 2: Sky Cyan / Blue
            (0, 215, 255),   # Rank 3: Golden Yellow
            (255, 0, 220),   # Rank 4: Neon Magenta
            (0, 140, 255),   # Rank 5+: Bright Orange
        ]

        top_k_info = []
        smooth_boxes_dict = {}
        for rank_idx, (tid, score) in enumerate(selected_tracks, 1):
            disp_sc = max(0.0, min(100.0, (score + 1.0) / 2.0 * 100.0))
            t_meta = tracklets_meta.get(tid, {})
            s_frame = t_meta.get("start_frame", 0)
            e_frame = t_meta.get("end_frame", total_frames - 1)
            out_cam = t_meta.get("out_of_camera", False)
            exit_f = t_meta.get("exit_frame", e_frame)

            raw_b = tracklets_boxes.get(tid, {})
            smooth_b = self._smooth_boxes(raw_b, alpha=0.75)
            smooth_boxes_dict[tid] = smooth_b

            color = RANK_COLORS[(rank_idx - 1) % len(RANK_COLORS)]
            top_k_info.append({
                "rank": rank_idx,
                "target_id": tid,
                "raw_score": score,
                "display_score": disp_sc,
                "start_frame": s_frame,
                "end_frame": e_frame,
                "out_of_camera": out_cam,
                "exit_frame": exit_f,
                "color": color,
            })

        best_info = top_k_info[0]
        best_tid = best_info["target_id"]
        best_score = best_info["raw_score"]
        display_score = best_info["display_score"]
        start_frame = best_info["start_frame"]
        end_frame = best_info["end_frame"]
        out_of_camera = best_info["out_of_camera"]
        exit_frame = best_info["exit_frame"]
        smooth_boxes = smooth_boxes_dict[best_tid]

        if progress_callback:
            if actual_k > 1:
                progress_callback(
                    0.72,
                    f"Đã chọn Top-{actual_k} mục tiêu (Top 1: ID #{best_tid} - {display_score:.1f}%). Đang xuất video..."
                )
            else:
                progress_callback(
                    0.72,
                    f"Mục tiêu: ID #{best_tid} ({display_score:.1f}%). Khung hình {start_frame} -> {end_frame}. Đang xuất video..."
                )

        # 5. Render Output Video
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        f_idx = 0
        while cap.isOpened() and f_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            in_view_ranks = []
            # Draw Top-K targets in reverse order so Top 1 appears on top
            for info in reversed(top_k_info):
                tid = info["target_id"]
                sm_boxes = smooth_boxes_dict[tid]
                if f_idx in sm_boxes:
                    in_view_ranks.append(info)
                    x1, y1, x2, y2 = sm_boxes[f_idx]
                    color = info["color"]

                    # Draw high-visibility bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Modern corner reticles
                    corner_len = min(22, max(8, (x2 - x1) // 4, (y2 - y1) // 4))
                    if corner_len > 0:
                        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 4)
                        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 4)
                        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 4)
                        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 4)
                        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 4)
                        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 4)
                        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 4)
                        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 4)

                    # Person badge label
                    if actual_k == 1:
                        badge_text = f"TARGET #{tid} ({info['display_score']:.1f}%)"
                    else:
                        badge_text = f"TOP #{info['rank']} [ID:{tid}] ({info['display_score']:.1f}%)"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.52
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)
                    label_y1 = max(0, y1 - th - 10)
                    label_y2 = y1
                    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 14, label_y2), color, -1)
                    cv2.putText(frame, badge_text, (x1 + 6, label_y2 - 6), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Header Status Banner
            header_h = 44
            cv2.rectangle(frame, (8, 8), (width - 8, 8 + header_h), (20, 20, 30), -1)
            cv2.rectangle(frame, (8, 8), (width - 8, 8 + header_h), (80, 80, 100), 1)

            if actual_k == 1:
                target_in_view = (f_idx in smooth_boxes)
                if target_in_view:
                    status_text = f"TARGET IN VIEW: #{best_tid} ({display_score:.1f}%)"
                    status_color = (0, 255, 127)  # Green
                elif out_of_camera and f_idx >= exit_frame:
                    status_text = f"TARGET EXITED CAMERA (Frame {exit_frame}/{total_frames})"
                    status_color = (0, 140, 255)  # Orange warning
                elif f_idx < start_frame:
                    status_text = f"WAITING FOR TARGET TO ENTER..."
                    status_color = (200, 200, 200)  # Gray
                else:
                    status_text = f"TRACKING IDLE"
                    status_color = (180, 180, 180)
            else:
                if in_view_ranks:
                    parts = [f"#{t['rank']}(ID:{t['target_id']}: {t['display_score']:.0f}%)" for t in in_view_ranks]
                    status_text = f"TOP-{actual_k} IN VIEW: " + ", ".join(parts)
                    status_color = (0, 255, 127)
                else:
                    status_text = f"TOP-{actual_k} TRACKING (WAITING FOR TARGETS)"
                    status_color = (180, 180, 180)

            # Query text
            q_short = query_text if len(query_text) <= 46 else query_text[:43] + "..."
            cv2.putText(frame, f"Query: \"{q_short}\"", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Status: {status_text}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 2, cv2.LINE_AA)

            # Frame Counter on top right
            frame_counter_str = f"[{f_idx + 1}/{total_frames}]"
            (fc_w, _), _ = cv2.getTextSize(frame_counter_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, frame_counter_str, (width - 18 - fc_w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 180), 1, cv2.LINE_AA)

            writer.write(frame)
            f_idx += 1

            if progress_callback and f_idx % 10 == 0:
                progress_callback(
                    0.72 + (f_idx / total_frames) * 0.27,
                    f"Đang xuất video có tracking box Top-{actual_k}... {f_idx}/{total_frames} frames"
                )

        cap.release()
        writer.release()

        if progress_callback:
            progress_callback(1.0, "Hoàn thành xuất video thành công!")

        best_crop = tracklets_crops[best_tid][0] if len(tracklets_crops[best_tid]) > 0 else None

        return {
            "target_id": best_tid,
            "raw_score": best_score,
            "display_score": display_score,
            "output_path": os.path.abspath(output_path),
            "total_persons_detected": len(tracklets_boxes),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "exit_frame": exit_frame,
            "out_of_camera": out_of_camera,
            "total_frames": total_frames,
            "tracked_frames_count": len(smooth_boxes),
            "best_crop": best_crop,
            "all_scores": track_scores,
            "top_k": actual_k,
            "top_k_results": top_k_info,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MFGF TVPR: Text-Guided Continuous Person Tracking CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", "-v", type=str, required=True, help="Đường dẫn đến file video đầu vào (.mp4)")
    parser.add_argument("--text", "-t", "--query", "-q", dest="text", type=str, required=True, help="Câu mô tả đối tượng cần theo vết (Tiếng Việt hoặc Tiếng Anh)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Đường dẫn lưu video kết quả đã vẽ bounding box")
    parser.add_argument("--top_k", "-k", type=int, default=1, help="Số lượng đối tượng hàng đầu cần theo vết (Top-K, từ 1 đến 5)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Đường dẫn file cấu hình YAML (default: auto)")
    parser.add_argument("--checkpoint", "-m", type=str, default=None, help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--max_frames", type=int, default=None, help="Giới hạn số khung hình xử lý (None: toàn bộ video)")

    args = parser.parse_args()

    print("=" * 70)
    print("MFGF TVPR - Text-Guided Continuous Person Tracking Pipeline (CLI)")
    print("=" * 70)
    print(f"[*] Video:      {args.video}")
    print(f"[*] Query Text: {args.text}")
    print(f"[*] Top-K:      {args.top_k}")
    print(f"[*] Output:     {args.output or 'Tự động lưu vào thư mục outputs/'}")
    print("=" * 70)

    engine = MFGFTrackingEngine(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )

    def print_progress(progress: float, message: str):
        pct = int(progress * 100)
        print(f"[{pct:3d}%] {message}", flush=True)

    result = engine.process_and_track(
        video_path=args.video,
        query_text=args.text,
        output_path=args.output,
        max_frames=args.max_frames,
        progress_callback=print_progress,
        top_k=args.top_k
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ THEO VẾT ĐỐI TƯỢNG (TRACKING REPORT):")
    print("=" * 70)
    print(f"[+] Target ID hàng đầu: #{result['target_id']}")
    print(f"[+] Độ tin cậy khớp:    {result['display_score']:.2f}% (Điểm thô: {result['raw_score']:.4f})")
    print(f"[+] Tổng số người quét: {result['total_persons_detected']}")
    print(f"[+] Khung hình bắt đầu: {result['start_frame']} -> kết thúc: {result['end_frame']}")
    if result["out_of_camera"]:
        print(f"[!] Trạng thái:        Đối tượng ĐÃ RỜI KHỎI KHUNG HÌNH tại frame {result['exit_frame']}")
    else:
        print(f"[+] Trạng thái:        Đối tượng xuất hiện đến hết video")
    print(f"[+] Video đã lưu tại:   {result['output_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
