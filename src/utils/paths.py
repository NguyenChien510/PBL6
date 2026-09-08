"""
Centralized Path Management for MFGF TVPR Project
Single Source of Truth cho toàn bộ đường dẫn trong dự án.
Tự động xác định PROJECT_ROOT và cung cấp các hàm phân giải thông minh,
giúp project chạy mượt mà từ cả thư mục gốc (D:\\PBL6) lẫn thư mục con (D:\\PBL6\\src).
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union


# 1. Xác định thư mục gốc của repository (PROJECT_ROOT)
# File này: src/utils/paths.py
# utils -> src -> PBL6 (PROJECT_ROOT)
_UTILS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _UTILS_DIR.parent
_CANDIDATE_ROOT = _SRC_DIR.parent

if (_CANDIDATE_ROOT / "dataset").exists() or (_CANDIDATE_ROOT / "data").exists() or (_CANDIDATE_ROOT / "src").exists():
    PROJECT_ROOT = _CANDIDATE_ROOT
else:
    # Fallback nếu cấu trúc thư mục bị thay đổi
    cwd = Path.cwd().resolve()
    if ((cwd / "dataset").exists() or (cwd / "data").exists()) and (cwd / "src").exists():
        PROJECT_ROOT = cwd
    elif ((cwd.parent / "dataset").exists() or (cwd.parent / "data").exists()) and (cwd.parent / "src").exists():
        PROJECT_ROOT = cwd.parent
    else:
        PROJECT_ROOT = _CANDIDATE_ROOT

# Đảm bảo src luôn có trong sys.path để các module import thuận tiện
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 2. Định nghĩa các đường dẫn chuẩn hóa trong dự án
# Tự động nhận diện thư mục dữ liệu (hỗ trợ cả folder 'dataset' lẫn 'data')
if (PROJECT_ROOT / "dataset").exists():
    DATA_DIR = PROJECT_ROOT / "dataset"
elif (PROJECT_ROOT / "data").exists():
    DATA_DIR = PROJECT_ROOT / "data"
else:
    DATA_DIR = PROJECT_ROOT / "dataset"

CAPTIONS_DIR = DATA_DIR / "captions"
VIDEOS_DIR = DATA_DIR / "videos"

CONFIGS_DIR = SRC_DIR / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"

CHECKPOINTS_DIR = SRC_DIR / "checkpoints"
BEST_CKPT_PATH = CHECKPOINTS_DIR / "best.pth"
LATEST_CKPT_PATH = CHECKPOINTS_DIR / "latest.pth"

OUTPUTS_DIR = SRC_DIR / "outputs"
REPORTS_DIR = SRC_DIR / "reports"


def ensure_project_dirs():
    """Tạo sẵn các thư mục lưu trữ cần thiết nếu chưa có."""
    for d in [DATA_DIR, CAPTIONS_DIR, VIDEOS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def resolve_data_root(data_root: Optional[Union[str, Path]] = None, dataset_name: str = "TVPReid-Combined") -> str:
    """
    Phân giải đường dẫn thư mục dữ liệu data_root.
    Ưu tiên:
      1. Nếu data_root là 'auto', None, rỗng, trỏ đến '/content', 'content', 'D:/AIBeginner/...'
         hoặc đường dẫn không tồn tại: tự động dùng DATA_DIR của dự án (PROJECT_ROOT / 'data').
      2. Nếu data_root là đường dẫn tương đối: kiểm tra relative theo CWD, PROJECT_ROOT, SRC_DIR.
      3. Nếu data_root là đường dẫn tuyệt đối tồn tại: trả về trực tiếp.
    """
    invalid_defaults = {"auto", "none", "null", "content", "/content"}

    if data_root is None:
        return str(DATA_DIR)

    raw_str = str(data_root).strip()
    if not raw_str or raw_str.lower() in invalid_defaults:
        return str(DATA_DIR)

    path_obj = Path(raw_str).expanduser()

    # Nếu đường dẫn truyền vào tồn tại
    if path_obj.exists():
        return str(path_obj.resolve())

    # Kiểm tra đường dẫn tương đối
    for base in [PROJECT_ROOT, SRC_DIR, Path.cwd()]:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return str(candidate)

    # Fallback về thư mục dữ liệu mặc định của project (hỗ trợ cả SSD Colab /content/dataset, dataset/ lẫn data/)
    for candidate in [Path("/content/dataset"), PROJECT_ROOT / "dataset", PROJECT_ROOT / "data", DATA_DIR]:
        if candidate.exists() and (candidate / "captions").exists():
            return str(candidate)
    for candidate in [Path("/content/dataset"), PROJECT_ROOT / "dataset", PROJECT_ROOT / "data", DATA_DIR]:
        if candidate.exists():
            return str(candidate)

    return str(path_obj)


def resolve_config_path(config_path: Optional[Union[str, Path]] = None) -> str:
    """
    Phân giải đường dẫn file cấu hình YAML.
    Tự động tìm kiếm nếu truyền đường dẫn tương đối như 'configs/default.yaml'.
    """
    if config_path is None or str(config_path).strip().lower() in {"", "auto", "none", "default"}:
        return str(DEFAULT_CONFIG_PATH)

    path_obj = Path(config_path).expanduser()
    if path_obj.exists():
        return str(path_obj.resolve())

    # Tìm kiếm tương đối từ CWD, PROJECT_ROOT, SRC_DIR
    for base in [PROJECT_ROOT, SRC_DIR, Path.cwd()]:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return str(candidate)

    # Tìm trong CONFIGS_DIR bằng tên file
    candidate_in_configs = CONFIGS_DIR / path_obj.name
    if candidate_in_configs.exists():
        return str(candidate_in_configs)

    if DEFAULT_CONFIG_PATH.exists():
        return str(DEFAULT_CONFIG_PATH)

    return str(path_obj)


def resolve_checkpoint_path(checkpoint_path: Optional[Union[str, Path]] = None, preferred: str = "best") -> str:
    """
    Phân giải đường dẫn file checkpoint (.pth).
    Tự động tìm kiếm trong CHECKPOINTS_DIR nếu truyền đường dẫn tương đối (ví dụ 'best.pth').
    """
    if preferred == "best":
        for cand in [CHECKPOINTS_DIR / "best.pth", CHECKPOINTS_DIR / "mfgf_best.pth", CHECKPOINTS_DIR / "latest.pth", CHECKPOINTS_DIR / "mfgf_latest.pth"]:
            if cand.exists():
                default_ckpt = cand
                break
        else:
            default_ckpt = BEST_CKPT_PATH
    else:
        for cand in [CHECKPOINTS_DIR / "latest.pth", CHECKPOINTS_DIR / "mfgf_latest.pth", CHECKPOINTS_DIR / "best.pth", CHECKPOINTS_DIR / "mfgf_best.pth"]:
            if cand.exists():
                default_ckpt = cand
                break
        else:
            default_ckpt = LATEST_CKPT_PATH

    if checkpoint_path is None or str(checkpoint_path).strip().lower() in {"", "auto", "none", "default"}:
        return str(default_ckpt)

    path_obj = Path(checkpoint_path).expanduser()
    if path_obj.exists():
        return str(path_obj.resolve())

    # Tìm kiếm tương đối
    for base in [PROJECT_ROOT, SRC_DIR, Path.cwd()]:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return str(candidate)

    # Tìm trong CHECKPOINTS_DIR bằng tên file
    candidate_in_ckpts = CHECKPOINTS_DIR / path_obj.name
    if candidate_in_ckpts.exists():
        return str(candidate_in_ckpts)

    # Fallback giữa tên có/không có tiền tố mfgf_
    alt_name = path_obj.name.replace("mfgf_", "") if path_obj.name.startswith("mfgf_") else f"mfgf_{path_obj.name}"
    alt_in_ckpts = CHECKPOINTS_DIR / alt_name
    if alt_in_ckpts.exists():
        return str(alt_in_ckpts)

    return str(path_obj)


def resolve_output_path(output_path: Optional[Union[str, Path]] = None, default_filename: str = "tracked_output.mp4") -> str:
    """
    Phân giải đường dẫn file kết quả xuất ra (video, biểu đồ...).
    Đảm bảo thư mục cha luôn được tạo sẵn.
    """
    if output_path is None or str(output_path).strip() == "":
        out_file = OUTPUTS_DIR / default_filename
    else:
        path_obj = Path(output_path).expanduser()
        if not path_obj.is_absolute():
            out_file = (OUTPUTS_DIR / path_obj.name) if path_obj.parent == Path(".") else (PROJECT_ROOT / path_obj)
        else:
            out_file = path_obj

    out_file.parent.mkdir(parents=True, exist_ok=True)
    return str(out_file)
