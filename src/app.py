"""
MFGF TVPR - Modern Desktop GUI Application for Text-to-Video Person Tracking
Features:
  - Robust Continuous Tracking (tracks until the end of the video or until the person leaves camera)
  - Interactive Embedded Video Player (Play / Pause / Scrubber / Speed Control)
  - Quick-select sample dataset videos
  - High-visibility Neon Bounding Box & Target Confidence Score
  - Out-of-Camera Detection Badge
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional, List, Dict
import cv2

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from pipeline_tracking import MFGFTrackingEngine
from utils.paths import (
    VIDEOS_DIR,
    LATEST_CKPT_PATH,
    BEST_CKPT_PATH,
    OUTPUTS_DIR,
    resolve_data_root,
    resolve_checkpoint_path,
    resolve_output_path,
)


class MFGFTrackingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MFGF TVPR - Text-Guided Continuous Person Tracking")
        self.root.geometry("1180x820")
        self.root.minsize(1080, 740)

        # Premium Dark Palette (Catppuccin inspired)
        self.bg_main = "#181825"
        self.bg_card = "#1e1e2e"
        self.bg_surface = "#252538"
        self.bg_input = "#313244"
        self.fg_text = "#cdd6f4"
        self.fg_sub = "#a6adc8"
        self.accent_blue = "#89b4fa"
        self.accent_green = "#a6e3a1"
        self.accent_red = "#f38ba8"
        self.accent_yellow = "#f9e2af"
        self.accent_cyan = "#94e2d5"

        self.root.configure(bg=self.bg_main)

        # Variables
        self.video_path_var = tk.StringVar(value="")
        default_ckpt = str(LATEST_CKPT_PATH if LATEST_CKPT_PATH.exists() else BEST_CKPT_PATH)
        self.ckpt_path_var = tk.StringVar(value=default_ckpt)
        self.query_var = tk.StringVar(value="Người phụ nữ mặc áo khoác trắng và quần xanh dương.")
        self.status_var = tk.StringVar(value="Sẵn sàng thực hiện.")
        self.continuous_mode_var = tk.BooleanVar(value=True)
        self.top_k_var = tk.IntVar(value=1)

        # Tracking Results
        self.engine: Optional[MFGFTrackingEngine] = None
        self.last_output_video: Optional[str] = None
        self.tracking_metadata: Optional[Dict[str, any]] = None

        # Video Player State
        self.player_cap: Optional[cv2.VideoCapture] = None
        self.player_playing = False
        self.player_total_frames = 0
        self.player_fps = 25.0
        self.player_current_frame = 0
        self.player_speed = 1.0
        self.player_photo = None
        self._player_timer_id = None

        self._setup_style()
        self._build_ui()
        self._auto_detect_initial_video()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=self.bg_input,
            background=self.accent_green,
            darkcolor=self.accent_green,
            lightcolor=self.accent_green,
            bordercolor=self.bg_card
        )

        style.configure(
            "Custom.Horizontal.TScale",
            troughcolor=self.bg_input,
            background=self.accent_blue,
            sliderlength=16,
            sliderrelief=tk.FLAT
        )

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.root, bg=self.bg_card, height=68)
        header.pack(fill=tk.X, padx=14, pady=(12, 8))

        title_box = tk.Frame(header, bg=self.bg_card)
        title_box.pack(side=tk.LEFT, padx=16, pady=8)

        title = tk.Label(
            title_box,
            text="🎯 MFGF TVPR — Text-to-Video Continuous Person Tracking",
            font=("Segoe UI", 15, "bold"),
            bg=self.bg_card,
            fg=self.accent_blue
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            title_box,
            text="Theo vết người liên tục theo mô tả tiếng Việt (Tự động bám sát đến hết video hoặc khi rời khỏi camera)",
            font=("Segoe UI", 9),
            bg=self.bg_card,
            fg=self.fg_sub
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # 2. Main Content Split: Left Controls (480px) & Right Video Player (expand)
        content_frame = tk.Frame(self.root, bg=self.bg_main)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

        left_col = tk.Frame(content_frame, bg=self.bg_card, width=490)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=0)
        left_col.pack_propagate(False)

        right_col = tk.Frame(content_frame, bg=self.bg_card)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=0)

        self._build_left_controls(left_col)
        self._build_right_player(right_col)

    def _build_left_controls(self, parent: tk.Frame):
        container = tk.Frame(parent, bg=self.bg_card, padx=14, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        # Section 1: Video Selection
        sec1 = tk.LabelFrame(
            container,
            text=" 1. Chọn Video Đầu Vào ",
            bg=self.bg_card,
            fg=self.accent_cyan,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=8
        )
        sec1.pack(fill=tk.X, pady=(0, 10))

        # Sample dropdown row
        sample_row = tk.Frame(sec1, bg=self.bg_card)
        sample_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(sample_row, text="Mẫu có sẵn:", bg=self.bg_card, fg=self.fg_sub, font=("Segoe UI", 9)).pack(side=tk.LEFT)

        self.sample_combobox = ttk.Combobox(sample_row, values=[], state="readonly", width=28, font=("Segoe UI", 9))
        self.sample_combobox.pack(side=tk.LEFT, padx=(6, 6))
        self.sample_combobox.bind("<<ComboboxSelected>>", self._on_sample_selected)

        # File picker row
        file_row = tk.Frame(sec1, bg=self.bg_card)
        file_row.pack(fill=tk.X)

        ent_video = tk.Entry(
            file_row,
            textvariable=self.video_path_var,
            bg=self.bg_input,
            fg=self.fg_text,
            insertbackground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9)
        )
        ent_video.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 6))

        btn_browse = tk.Button(
            file_row,
            text="📂 Duyệt...",
            bg=self.bg_surface,
            fg=self.fg_text,
            activebackground=self.bg_input,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=10,
            command=self._browse_video
        )
        btn_browse.pack(side=tk.RIGHT)

        # Section 2: Query Text Description
        sec2 = tk.LabelFrame(
            container,
            text=" 2. Mô Tả Người Cần Tìm Kiếm (Tiếng Việt) ",
            bg=self.bg_card,
            fg=self.accent_cyan,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=8
        )
        sec2.pack(fill=tk.X, pady=(0, 10))

        ent_query = tk.Entry(
            sec2,
            textvariable=self.query_var,
            bg=self.bg_input,
            fg=self.fg_text,
            insertbackground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10)
        )
        ent_query.pack(fill=tk.X, ipady=5, pady=(0, 6))

        # Preset buttons
        preset_frame = tk.Frame(sec2, bg=self.bg_card)
        preset_frame.pack(fill=tk.X)

        presets = [
            ("Áo trắng, quần xanh", "Người phụ nữ mặc áo khoác trắng và quần xanh."),
            ("Áo tối màu", "Người đàn ông mặc áo khoác tối màu và quần đen."),
            ("Áo xanh, túi xách", "Người phụ nữ mặc áo khoác màu xanh và mang túi xách."),
            ("Áo nâu, giày đỏ", "Người đàn ông mặc áo màu nâu và đi giày màu đỏ.")
        ]
        for p_label, p_text in presets:
            btn = tk.Button(
                preset_frame,
                text=p_label,
                bg=self.bg_surface,
                fg=self.fg_sub,
                activebackground=self.bg_input,
                font=("Segoe UI", 8),
                relief=tk.FLAT,
                padx=5,
                pady=1,
                command=lambda t=p_text: self.query_var.set(t)
            )
            btn.pack(side=tk.LEFT, padx=(0, 4), pady=2)

        # Section 3: Tracking Options
        sec3 = tk.LabelFrame(
            container,
            text=" 3. Cấu Hình Bám Sát & Checkpoint ",
            bg=self.bg_card,
            fg=self.accent_cyan,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=6
        )
        sec3.pack(fill=tk.X, pady=(0, 12))

        # Top-K Selector Row
        topk_row = tk.Frame(sec3, bg=self.bg_card)
        topk_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            topk_row,
            text="Số người Top-K xuất hiện:",
            bg=self.bg_card,
            fg=self.fg_text,
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        self.topk_combobox = ttk.Combobox(
            topk_row,
            textvariable=self.top_k_var,
            values=[1, 2, 3, 5, 10],
            state="readonly",
            width=5,
            font=("Segoe UI", 9, "bold")
        )
        self.topk_combobox.pack(side=tk.LEFT, padx=(8, 6))
        self.topk_combobox.current(0)

        tk.Label(
            topk_row,
            text="(Hiển thị K mục tiêu khớp nhất)",
            bg=self.bg_card,
            fg=self.fg_sub,
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT)

        cb_continuous = tk.Checkbutton(
            sec3,
            text="Bám sát liên tục đến hết video (hoặc dừng khi người ra khỏi camera)",
            variable=self.continuous_mode_var,
            bg=self.bg_card,
            fg=self.accent_green,
            selectcolor=self.bg_input,
            activebackground=self.bg_card,
            activeforeground=self.accent_green,
            font=("Segoe UI", 9, "bold")
        )
        cb_continuous.pack(anchor="w", pady=(0, 4))

        ckpt_row = tk.Frame(sec3, bg=self.bg_card)
        ckpt_row.pack(fill=tk.X)
        tk.Label(ckpt_row, text="Model Checkpoint:", bg=self.bg_card, fg=self.fg_sub, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ent_ckpt = tk.Entry(
            ckpt_row,
            textvariable=self.ckpt_path_var,
            bg=self.bg_input,
            fg=self.fg_text,
            relief=tk.FLAT,
            font=("Segoe UI", 8)
        )
        ent_ckpt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Action Run Button
        self.btn_run = tk.Button(
            container,
            text="🚀 BẮT ĐẦU THEO VẾT LIÊN TỤC (TRACKING)",
            bg=self.accent_green,
            fg="#11111b",
            activebackground="#94e2d5",
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            pady=8,
            command=self._start_tracking_thread
        )
        self.btn_run.pack(fill=tk.X, pady=(0, 8))

        # Progress bar & Status
        self.progress_bar = ttk.Progressbar(
            container,
            style="Custom.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 4))

        self.lbl_status = tk.Label(
            container,
            textvariable=self.status_var,
            bg=self.bg_card,
            fg=self.accent_blue,
            font=("Segoe UI", 9, "italic")
        )
        self.lbl_status.pack(anchor="w", pady=(0, 10))

        # Section 4: Result Statistics Card
        self.stats_card = tk.LabelFrame(
            container,
            text=" 4. Thống Kê Theo Vết (Tracking Stats) ",
            bg=self.bg_card,
            fg=self.accent_green,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=8
        )
        self.stats_card.pack(fill=tk.BOTH, expand=True)

        self.lbl_target_badge = tk.Label(
            self.stats_card,
            text="⚪ CHƯA CHẠY",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_input,
            fg=self.fg_sub,
            padx=10,
            pady=4
        )
        self.lbl_target_badge.pack(anchor="w", pady=(0, 6))

        self.lbl_stats_info = tk.Label(
            self.stats_card,
            text="Chọn video và nhấn nút Bắt Đầu để thực hiện theo vết người.",
            bg=self.bg_card,
            fg=self.fg_text,
            font=("Segoe UI", 9),
            justify=tk.LEFT
        )
        self.lbl_stats_info.pack(anchor="w", pady=(0, 8))

        # External Open Buttons Row
        btn_ext_row = tk.Frame(self.stats_card, bg=self.bg_card)
        btn_ext_row.pack(anchor="w", pady=(4, 0))

        self.btn_ext_player = tk.Button(
            btn_ext_row,
            text="▶ Mở Windows Player",
            bg=self.bg_surface,
            fg=self.accent_blue,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            state=tk.DISABLED,
            command=self._open_external_player
        )
        self.btn_ext_player.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_ext_folder = tk.Button(
            btn_ext_row,
            text="📁 Mở Thư Mục",
            bg=self.bg_surface,
            fg=self.fg_text,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            state=tk.DISABLED,
            command=self._open_output_folder
        )
        self.btn_ext_folder.pack(side=tk.LEFT)

    def _build_right_player(self, parent: tk.Frame):
        container = tk.Frame(parent, bg=self.bg_card, padx=14, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        # Player Header & Badge
        player_top = tk.Frame(container, bg=self.bg_card)
        player_top.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            player_top,
            text="📺 Khung Chiếu Video Trực Tiếp (Live Video Preview)",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_card,
            fg=self.accent_blue
        ).pack(side=tk.LEFT)

        self.lbl_player_status = tk.Label(
            player_top,
            text="Sẵn sàng",
            font=("Segoe UI", 9, "bold"),
            bg=self.bg_surface,
            fg=self.accent_cyan,
            padx=8,
            pady=2
        )
        self.lbl_player_status.pack(side=tk.RIGHT)

        # Video Canvas Container
        self.canvas_frame = tk.Frame(container, bg="#0d0d15", relief=tk.SOLID, bd=1)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d0d15", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Video Controls Bar
        ctrl_bar = tk.Frame(container, bg=self.bg_surface, padx=10, pady=8)
        ctrl_bar.pack(fill=tk.X)

        # Row 1: Scrubber Scale & Time label
        scrub_row = tk.Frame(ctrl_bar, bg=self.bg_surface)
        scrub_row.pack(fill=tk.X, pady=(0, 6))

        self.scale_scrub = ttk.Scale(
            scrub_row,
            style="Custom.Horizontal.TScale",
            orient="horizontal",
            from_=0,
            to=100,
            command=self._on_scrub_change
        )
        self.scale_scrub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.lbl_time = tk.Label(
            scrub_row,
            text="0 / 0 frames (0.0s)",
            font=("Segoe UI", 9),
            bg=self.bg_surface,
            fg=self.fg_sub
        )
        self.lbl_time.pack(side=tk.RIGHT)

        # Row 2: Play, Pause, Replay, Speed buttons
        btn_row = tk.Frame(ctrl_bar, bg=self.bg_surface)
        btn_row.pack(fill=tk.X)

        self.btn_play_pause = tk.Button(
            btn_row,
            text="▶ Phát Video",
            bg=self.accent_blue,
            fg="#11111b",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=2,
            command=self._toggle_play_pause
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_replay = tk.Button(
            btn_row,
            text="↺ Phát Lại",
            bg=self.bg_input,
            fg=self.fg_text,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=10,
            pady=2,
            command=self._replay_video
        )
        self.btn_replay.pack(side=tk.LEFT, padx=(0, 12))

        # Speed buttons
        tk.Label(btn_row, text="Tốc độ:", bg=self.bg_surface, fg=self.fg_sub, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 4))
        for spd in [0.5, 1.0, 1.5, 2.0]:
            b = tk.Button(
                btn_row,
                text=f"{spd}x",
                bg=self.bg_input,
                fg=self.fg_text,
                font=("Segoe UI", 8),
                relief=tk.FLAT,
                padx=5,
                pady=1,
                command=lambda s=spd: self._set_speed(s)
            )
            b.pack(side=tk.LEFT, padx=2)

    def _auto_detect_initial_video(self):
        """Discovers sample dataset videos and populates sample combobox."""
        candidate_dirs = [
            VIDEOS_DIR,
            os.path.join(resolve_data_root(), "videos"),
            resolve_data_root()
        ]
        found_videos = []
        target_dir = None
        for sdir in candidate_dirs:
            sdir_str = str(sdir)
            if os.path.exists(sdir_str):
                files = [f for f in os.listdir(sdir_str) if f.endswith(".mp4")]
                if files:
                    target_dir = sdir_str
                    found_videos = sorted(files)[:30]
                    break

        if found_videos and target_dir:
            self.sample_dir = target_dir
            self.sample_combobox["values"] = found_videos
            self.sample_combobox.current(0)
            initial_path = os.path.join(target_dir, found_videos[0])
            self.video_path_var.set(initial_path)
            self._load_video_preview(initial_path)

    def _on_sample_selected(self, event=None):
        selected_file = self.sample_combobox.get()
        if hasattr(self, "sample_dir") and selected_file:
            path = os.path.join(self.sample_dir, selected_file)
            self.video_path_var.set(path)
            self._load_video_preview(path)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Chọn Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if path:
            self.video_path_var.set(path)
            self._load_video_preview(path)

    def _start_tracking_thread(self):
        video_path = self.video_path_var.get().strip()
        ckpt_path = self.ckpt_path_var.get().strip()
        query_text = self.query_var.get().strip()
        top_k = int(self.top_k_var.get()) if hasattr(self, "top_k_var") else 1

        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Lỗi", "Vui lòng chọn một file video hợp lệ!")
            return

        if not query_text:
            messagebox.showerror("Lỗi", "Vui lòng nhập câu mô tả người cần tìm!")
            return

        self._stop_video_playback()
        self.btn_run.config(state=tk.DISABLED, text="⏳ ĐANG XỬ LÝ THEO VẾT...", bg=self.accent_yellow)
        self.btn_ext_player.config(state=tk.DISABLED)
        self.btn_ext_folder.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0

        # Run pipeline in background thread
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(video_path, ckpt_path, query_text, top_k),
            daemon=True
        )
        thread.start()

    def _run_pipeline(self, video_path: str, ckpt_path: str, query_text: str, top_k: int = 1):
        try:
            def update_progress(ratio: float, status_msg: str):
                def _gui_update():
                    self.progress_bar["value"] = int(ratio * 100)
                    self.status_var.set(status_msg)
                self.root.after(0, _gui_update)

            # Initialize engine once
            if self.engine is None or getattr(self, "_current_ckpt", None) != ckpt_path:
                update_progress(0.02, "Đang nạp mô hình MFGF...")
                self.engine = MFGFTrackingEngine(checkpoint_path=ckpt_path)
                self._current_ckpt = ckpt_path

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"tracked_{base_name}.mp4"
            output_path = resolve_output_path(output_filename)

            result = self.engine.process_and_track(
                video_path=video_path,
                query_text=query_text,
                output_path=output_path,
                max_frames=None,  # Full video tracking!
                progress_callback=update_progress,
                top_k=top_k
            )

            self.last_output_video = result["output_path"]
            self.tracking_metadata = result

            # Update GUI upon completion
            def _on_success():
                total_f = result["total_frames"]
                tracked_f = result["tracked_frames_count"]
                out_cam = result["out_of_camera"]
                exit_f = result["exit_frame"]
                score = result["display_score"]
                tid = result["target_id"]
                top_k_results = result.get("top_k_results", [])
                actual_k = result.get("top_k", 1)

                if actual_k > 1 and len(top_k_results) > 1:
                    self.lbl_target_badge.config(
                        text=f"🟢 ĐÃ BÁM SÁT TOP-{len(top_k_results)} MỤC TIÊU ({total_f} frames)",
                        bg="#254530",
                        fg=self.accent_green
                    )
                    stats_lines = [
                        f"★ TOP-{len(top_k_results)} MỤC TIÊU KHỚP NHẤT TRONG VIDEO:\n"
                    ]
                    for item in top_k_results:
                        r = item["rank"]
                        t_id = item["target_id"]
                        sc = item["display_score"]
                        sf = item["start_frame"]
                        ef = item["end_frame"]
                        stats_lines.append(f"  • Top #{r}: ID #{t_id}  |  Độ khớp: {sc:.1f}%  |  Khung hình {sf} -> {ef}")
                    stats_lines.append(f"\n📁 File video xuất: {os.path.basename(result['output_path'])}")
                    stats_text = "\n".join(stats_lines)
                    self.status_var.set(f"Hoàn thành! Đã bám sát Top-{len(top_k_results)} mục tiêu. Đang phát video...")
                else:
                    # Badge update for single target
                    if out_cam:
                        self.lbl_target_badge.config(
                            text=f"🔴 RỜI KHỎI CAMERA (Khung hình {exit_f}/{total_f})",
                            bg="#452530",
                            fg=self.accent_red
                        )
                        camera_status_str = f"Người đã đi ra khỏi tầm quan sát của camera ở khung hình {exit_f}."
                    else:
                        self.lbl_target_badge.config(
                            text=f"🟢 BÁM SÁT TOÀN BỘ VIDEO ({tracked_f}/{total_f} frames)",
                            bg="#254530",
                            fg=self.accent_green
                        )
                        camera_status_str = f"Người xuất hiện và được bám sát liên tục đến hết video ({tracked_f}/{total_f} frames)."

                    stats_text = (
                        f"✓ Nhận diện mục tiêu: ID #{tid}\n"
                        f"★ Độ tin cậy khớp (Similarity): {score:.1f}%\n"
                        f"⏱ Thời lượng xuất hiện: Khung hình {result['start_frame']} -> {result['end_frame']}\n"
                        f"📌 Trạng thái: {camera_status_str}\n"
                        f"📁 File xuất: {os.path.basename(result['output_path'])}"
                    )
                    self.status_var.set(f"Hoàn thành! Đã bám sát mục tiêu ({score:.1f}%). Đang phát video...")

                self.lbl_stats_info.config(text=stats_text)
                self.btn_run.config(state=tk.NORMAL, text="🚀 BẮT ĐẦU THEO VẾT LIÊN TỤC (TRACKING)", bg=self.accent_green)
                self.btn_ext_player.config(state=tk.NORMAL)
                self.btn_ext_folder.config(state=tk.NORMAL)

                # Automatically load and play tracked video in built-in player
                self._load_video_preview(result["output_path"], auto_play=True)

            self.root.after(0, _on_success)

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_str = str(e)
            def _on_error(msg=err_str):
                self.status_var.set(f"Đã xảy ra lỗi: {msg}")
                self.btn_run.config(state=tk.NORMAL, text="🚀 BẮT ĐẦU THEO VẾT LIÊN TỤC (TRACKING)", bg=self.accent_green)
                messagebox.showerror("Lỗi Thực Thi", f"Lỗi: {msg}")
            self.root.after(0, _on_error)

    # =========================================================================
    # Video Player Methods
    # =========================================================================
    def _load_video_preview(self, video_path: str, auto_play: bool = False):
        self._stop_video_playback()
        if not os.path.exists(video_path):
            return

        self.player_cap = cv2.VideoCapture(video_path)
        if not self.player_cap.isOpened():
            return

        self.player_total_frames = int(self.player_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.player_fps = self.player_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.player_current_frame = 0

        self.scale_scrub.config(to=max(1, self.player_total_frames - 1))
        self.scale_scrub.set(0)
        self.lbl_player_status.config(text=f"Video: {os.path.basename(video_path)}")

        # Render first frame
        self._show_frame(0)

        if auto_play:
            self._start_video_playback()

    def _show_frame(self, frame_number: int):
        if self.player_cap is None or not self.player_cap.isOpened():
            return

        self.player_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.player_cap.read()
        if not ret:
            return

        self.player_current_frame = frame_number

        # Update time label
        cur_sec = frame_number / max(self.player_fps, 1.0)
        tot_sec = self.player_total_frames / max(self.player_fps, 1.0)
        self.lbl_time.config(text=f"Frame: {frame_number + 1} / {self.player_total_frames} ({cur_sec:.1f}s / {tot_sec:.1f}s)")

        # Render on Canvas with aspect ratio preservation
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if HAS_PIL:
            img = Image.fromarray(rgb)
            self.player_photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            x_pos = (cw - nw) // 2
            y_pos = (ch - nh) // 2
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.player_photo)

    def _play_loop(self):
        if not self.player_playing or self.player_cap is None:
            return

        next_f = self.player_current_frame + 1
        if next_f >= self.player_total_frames:
            # End of video reached
            self._stop_video_playback()
            self._show_frame(self.player_total_frames - 1)
            self.scale_scrub.set(self.player_total_frames - 1)
            return

        self._show_frame(next_f)
        self.scale_scrub.set(next_f)

        delay_ms = max(10, int((1000.0 / self.player_fps) / max(0.25, self.player_speed)))
        self._player_timer_id = self.root.after(delay_ms, self._play_loop)

    def _toggle_play_pause(self):
        if self.player_playing:
            self._stop_video_playback()
        else:
            if self.player_current_frame >= self.player_total_frames - 1:
                self.player_current_frame = 0
            self._start_video_playback()

    def _start_video_playback(self):
        if self.player_cap is not None and self.player_cap.isOpened():
            self.player_playing = True
            self.btn_play_pause.config(text="⏸ Tạm Dừng", bg=self.accent_yellow)
            self._play_loop()

    def _stop_video_playback(self):
        self.player_playing = False
        if self._player_timer_id is not None:
            self.root.after_cancel(self._player_timer_id)
            self._player_timer_id = None
        self.btn_play_pause.config(text="▶ Phát Video", bg=self.accent_blue)

    def _replay_video(self):
        self._stop_video_playback()
        self._show_frame(0)
        self.scale_scrub.set(0)
        self._start_video_playback()

    def _on_scrub_change(self, val):
        f = int(float(val))
        if f != self.player_current_frame:
            self._show_frame(f)

    def _set_speed(self, speed: float):
        self.player_speed = speed
        self.lbl_player_status.config(text=f"Tốc độ phát: {speed}x")

    def _on_canvas_resize(self, event=None):
        if self.player_cap is not None and not self.player_playing:
            self._show_frame(self.player_current_frame)

    def _open_external_player(self):
        if self.last_output_video and os.path.exists(self.last_output_video):
            if sys.platform == "win32":
                os.startfile(self.last_output_video)
            else:
                os.system(f"xdg-open '{self.last_output_video}'")

    def _open_output_folder(self):
        folder = os.path.abspath("outputs")
        if os.path.exists(folder):
            if sys.platform == "win32":
                os.system(f"explorer '{folder}'")
            else:
                os.system(f"xdg-open '{folder}'")


if __name__ == "__main__":
    root = tk.Tk()
    app = MFGFTrackingApp(root)
    root.mainloop()
