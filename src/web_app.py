"""
MFGF TVPR - Modern Gradio Web Application for Text-to-Video Person Tracking
Designed for both local execution and Docker container deployment.
Allows users to upload or select a video, enter a Vietnamese query text,
and watch the continuous person tracking result with Neon bounding boxes and metadata.
"""

import os
import sys
from pathlib import Path
import time
from typing import Optional, List, Tuple
import gradio as gr

# Đảm bảo src có trong sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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

# Global engine cache
_ENGINE_CACHE = {}


def get_engine(checkpoint_preference: str = "latest"):
    """Khởi tạo hoặc lấy engine từ bộ nhớ đệm (singleton cache)."""
    if checkpoint_preference not in _ENGINE_CACHE:
        ckpt = str(BEST_CKPT_PATH) if checkpoint_preference == "best" else str(LATEST_CKPT_PATH)
        print(f"[*] Đang khởi tạo MFGFTrackingEngine với checkpoint ({checkpoint_preference}): {ckpt}...", flush=True)
        _ENGINE_CACHE[checkpoint_preference] = MFGFTrackingEngine(
            checkpoint_path=ckpt
        )
    return _ENGINE_CACHE[checkpoint_preference]


def list_sample_videos() -> List[Tuple[str, str]]:
    """Tìm các video mẫu có sẵn trong thư mục dataset/videos."""
    samples = []
    if VIDEOS_DIR.exists():
        for p in sorted(VIDEOS_DIR.glob("*.mp4"))[:12]:
            samples.append((p.name, str(p.resolve())))
    return samples


def process_tracking(
    video_file,
    sample_video_choice,
    query_text: str,
    top_k: int,
    checkpoint_choice: str,
    max_frames: Optional[int],
    progress=gr.Progress(track_tqdm=True)
):
    """Xử lý theo vết đối tượng qua văn bản bằng MFGF/DINOv2 Engine."""
    if not query_text or not query_text.strip():
        raise gr.Error("Vui lòng nhập câu mô tả người cần theo vết!")

    # Xác định đường dẫn video
    video_path = None
    if video_file is not None:
        video_path = video_file
    elif sample_video_choice:
        video_path = sample_video_choice

    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Vui lòng tải lên video hoặc chọn 1 video mẫu từ danh sách!")

    progress(0.02, desc="Đang chuẩn bị mô hình...")
    start_time = time.time()

    try:
        engine = get_engine(checkpoint_choice)
    except Exception as e:
        raise gr.Error(f"Lỗi khi nạp mô hình: {str(e)}")

    def update_progress(p: float, msg: str):
        progress(p, desc=msg)

    out_name = f"web_tracked_{int(time.time())}.mp4"
    out_path = resolve_output_path(out_name)

    max_f = int(max_frames) if max_frames and max_frames > 0 else None

    try:
        result = engine.process_and_track(
            video_path=video_path,
            query_text=query_text.strip(),
            output_path=out_path,
            max_frames=max_f,
            progress_callback=update_progress,
            top_k=int(top_k)
        )
    except Exception as e:
        raise gr.Error(f"Lỗi trong quá trình theo vết: {str(e)}")

    elapsed = time.time() - start_time
    res_video_path = result["output_path"]

    # Xây dựng báo cáo kết quả Markdown
    target_id = result["target_id"]
    display_score = result["display_score"]
    total_persons = result["total_persons_detected"]
    total_frames = result["total_frames"]
    tracked_frames = result["tracked_frames_count"]

    if result["out_of_camera"]:
        status_badge = f"⚠️ **Rời khỏi camera** (Khung hình {result['exit_frame']}/{total_frames})"
    else:
        status_badge = "✅ **Trong khung hình** (Xuất hiện liên tục)"

    top_k_details = ""
    if "top_k_results" in result and len(result["top_k_results"]) > 1:
        top_k_details = "\n### Bảng xếp hạng Top-K:\n"
        for t in result["top_k_results"]:
            top_k_details += f"- **Hạng #{t['rank']}**: Person ID #{t['target_id']} - Độ tương đồng: `{t['display_score']:.2f}%`\n"

    report_md = f"""
### 🎯 Báo Cáo Kết Quả Theo Vết
- **Đối tượng mục tiêu số 1:** `Person #{target_id}`
- **Độ tin cậy ngữ nghĩa:** `{display_score:.2f}%`
- **Trạng thái đối tượng:** {status_badge}
- **Tổng số người xuất hiện:** `{total_persons}` người
- **Số khung hình theo vết:** `{tracked_frames}/{total_frames}` frames
- **Thời gian xử lý:** `{elapsed:.2f} giây`
{top_k_details}
"""

    return res_video_path, report_md


def create_ui():
    sample_videos = list_sample_videos()
    sample_choices = {name: path for name, path in sample_videos}

    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px 10px;
        background: linear-gradient(135deg, #1e1e2e 0%, #252538 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #313244;
    }
    .main-header h1 {
        color: #89b4fa;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .main-header p {
        color: #a6adc8;
        font-size: 15px;
        margin: 0;
    }
    """

    with gr.Blocks(title="MFGF TVPR - Text-Guided Continuous Person Tracking", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["main-header"]):
            gr.Markdown(
                """
                # 🔍 MFGF TVPR: Text-Guided Continuous Person Tracking
                **Hệ thống tìm kiếm và theo vết đối tượng người trong video bằng câu mô tả tiếng Việt**
                *Kết hợp Meta DINOv2 Foundation Vision & VinAI PhoBERT v2 Language Adapter*
                """
            )

        with gr.Row():
            # Cột trái: Đầu vào
            with gr.Column(scale=1):
                gr.Markdown("### 📥 1. Chọn hoặc Tải Video")
                video_input = gr.Video(label="Tải lên Video (.mp4)", sources=["upload"])

                sample_dropdown = gr.Dropdown(
                    label="Hoặc chọn nhanh video mẫu từ tập dữ liệu",
                    choices=list(sample_choices.keys()),
                    value=list(sample_choices.keys())[0] if sample_choices else None,
                    interactive=True
                )

                selected_path_state = gr.State(
                    value=sample_choices[list(sample_choices.keys())[0]] if sample_choices else None
                )

                def on_sample_change(choice):
                    return sample_choices.get(choice, None)

                sample_dropdown.change(on_sample_change, inputs=[sample_dropdown], outputs=[selected_path_state])

                gr.Markdown("### ✍️ 2. Mô tả Người Cần Tìm")
                query_box = gr.Textbox(
                    label="Câu mô tả đối tượng (Tiếng Việt)",
                    placeholder="ví dụ: người phụ nữ mặc áo đen đeo ba lô, người đàn ông mặc áo xanh...",
                    lines=2,
                    value="người phụ nữ mặc áo đen mang túi xách"
                )

                gr.Examples(
                    examples=[
                        ["người phụ nữ mặc áo đen mang túi xách"],
                        ["người đàn ông mặc áo xanh quần dài"],
                        ["người mặc áo sơ mi trắng"],
                        ["người mặc áo khoác đỏ"],
                    ],
                    inputs=[query_box],
                    label="Gợi ý câu truy vấn nhanh"
                )

                with gr.Accordion("⚙️ Cấu hình nâng cao", open=False):
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Top-K đối tượng theo vết (Vẽ khung cho K người có điểm cao nhất)"
                    )
                    checkpoint_radio = gr.Radio(
                        choices=["best", "latest"],
                        value="best" if BEST_CKPT_PATH.exists() else "latest",
                        label="Checkpoint mô hình"
                    )
                    max_frames_slider = gr.Slider(
                        minimum=0, maximum=1000, value=0, step=50,
                        label="Giới hạn frames (0 = xử lý toàn bộ video)"
                    )

                track_btn = gr.Button("🚀 BẮT ĐẦU THEO VẾT ĐỐI TƯỢNG", variant="primary", size="lg")

            # Cột phải: Kết quả đầu ra
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 3. Video Kết Quả Theo Vết")
                video_output = gr.Video(label="Kết quả theo vết (Bounding Box Neon & Score)", interactive=False)
                report_output = gr.Markdown(value="*Kết quả phân tích chi tiết sẽ hiển thị tại đây sau khi chạy.*")

        track_btn.click(
            fn=process_tracking,
            inputs=[
                video_input,
                selected_path_state,
                query_box,
                top_k_slider,
                checkpoint_radio,
                max_frames_slider,
            ],
            outputs=[
                video_output,
                report_output,
            ]
        )

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MFGF TVPR Gradio Web Application")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Địa chỉ IP lắng nghe (mặc định: 0.0.0.0)")
    parser.add_argument("--server-port", type=int, default=7860, help="Cổng chạy Web UI (mặc định: 7860)")
    parser.add_argument("--share", action="store_true", help="Tạo link Gradio Public Share")
    args = parser.parse_args()

    demo = create_ui()
    print(f"\n[+] Khởi động Web App tại: http://localhost:{args.server_port} (IP: {args.server_name})\n", flush=True)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )


if __name__ == "__main__":
    main()
