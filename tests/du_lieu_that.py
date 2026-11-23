"""Đọc video thật cho test (QĐ-31, QĐ-32: test chỉ dùng video thật, và chỉ dùng 4 video nhóm đưa).

Video là bộ EPFL phòng thí nghiệm 4 người (4p-c0 … c3.avi) trong data/ (không có trên GitHub, xem
data/README.md). Bộ này CHƯA có đáp án, nên test kiểm tra những điều đo được mà không cần đáp án:
khung tìm được có đúng cỡ người theo file hiệu chỉnh camera, số tên thừa ít, cùng một người giống màu
hơn hai người khác nhau… Các khung người dùng trong test được chọn bằng mắt trên frame thật (ghi rõ ở
từng hằng số).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.common.utils import camera_video

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
FEATURES_CFG = CONFIG["features"]
LAB4P = ROOT / "data/canh/epfl_lab4p_c0_c1"
LAB4P_CALIBRATION = ROOT / "data/goc/epfl_lab4p/calibration-4p.txt"
LAB4P_VIDEOS = [ROOT / f"data/goc/epfl_lab4p/4p-c{camera}.avi" for camera in range(4)]

# Khung người chọn bằng mắt trên camera 0 (cam_a của LAB4P), mỗi khung đúng một người:
MAN_BLACK_WHITE_700 = (121, 18, 63, 228)    # frame 700: đàn ông áo đen, quần trắng
WOMAN_BLACK_700 = (279, 17, 55, 178)        # frame 700: phụ nữ tóc dài, đồ đen
MAN_BLACK_WHITE_710 = (86, 20, 80, 226)     # frame 710: cùng người đàn ông áo đen, quần trắng
BLACK_WHITE_800 = (42, 0, 42, 155)          # frame 800: người áo đen, quần trắng
BROWN_800 = (126, 30, 60, 154)              # frame 800: người áo nâu
BROWN_805 = (121, 0, 93, 188)               # frame 805: cùng người áo nâu
FLAT_REGION_0 = (160, 152, 40, 120)         # frame 0: vùng sàn trơn nhất (độ lệch chuẩn xám ≈ 6)


def need(path: Path) -> Path:
    """Trả về đường dẫn; bỏ qua test nếu máy chưa có dữ liệu đó."""
    exists = camera_video(path / "cam_a") is not None if path.is_dir() or not path.suffix else path.exists()
    if not exists:
        pytest.skip(f"Thiếu dữ liệu thật {path.relative_to(ROOT)} (xem data/README.md)")
    return path


def read_frame(scene: Path, camera: str, index: int) -> np.ndarray:
    """Một frame của video thật."""
    capture = cv2.VideoCapture(str(camera_video(scene / camera)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    assert ok, f"Không đọc được frame {index} của {scene.name}/{camera}"
    return frame


def crop(frame: np.ndarray, box) -> np.ndarray:
    """Phần ảnh nằm trong box (x, y, w, h)."""
    x, y, w, h = box
    return frame[max(0, y):y + h, max(0, x):x + w]
