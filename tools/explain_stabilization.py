"""Ảnh minh họa cho bước ổn định hình (src/detect/stabilization.py, QĐ-30).

Cách dùng:
    python -m tools.explain_stabilization --video <video có camera bị trôi> --frame 300 --out outputs/explain/on_dinh_hinh

Video hiện có của nhóm quay bằng camera cố định (độ trôi gần 0); công cụ dành cho video quay cầm tay hoặc
chân máy không chắc.

Ghi ra hai ảnh:
1. `1_do_troi.png`: camera trôi bao nhiêu pixel so với frame đầu (dx, dy) theo thời gian. Camera đứng
   yên thì hai đường nằm sát 0.
2. `2_mat_na.png`: cùng một frame, mặt nạ foreground KHÔNG ổn định hình (trái) và CÓ ổn định hình (phải).
   Không ổn định thì mép cửa sổ, cột, lan can bị coi là "đang chuyển động".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.utils import iter_video  # noqa: E402
from src.detect.detection import PersonDetector  # noqa: E402
from src.detect.stabilization import Stabilizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(video: Path, config: dict, frame_index: int):
    """Chạy song song hai bộ tìm người (không / có ổn định hình); trả về độ trôi và hai mặt nạ ở frame_index."""
    stabilizer = Stabilizer(config["stabilization"])
    plain, stable = PersonDetector(config["detection"]), PersonDetector(config["detection"])
    drift, masks = [], None
    for index, frame in iter_video(video):
        work = stabilizer.apply(frame)
        drift.append(-stabilizer.matrix[:, 2])
        plain.detect(frame)
        stable.detect(work, stabilizer.valid_mask)
        if index == frame_index:
            masks = (frame, plain.last_mask.copy(), stable.last_mask.copy())
        if index >= frame_index and index >= 300:
            break
    return np.array(drift), masks


def plot_drift(drift: np.ndarray, out: Path) -> None:
    """Độ trôi dx, dy (pixel) so với frame đầu theo từng frame."""
    plt.figure(figsize=(8, 4))
    plt.plot(drift[:, 0], label="dx (ngang)")
    plt.plot(drift[:, 1], label="dy (dọc)")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.xlabel("frame")
    plt.ylabel("camera trôi so với frame đầu (pixel)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def draw_masks(frame: np.ndarray, plain: np.ndarray, stable: np.ndarray, out: Path) -> None:
    """Hai mặt nạ foreground phủ đỏ lên frame, đặt cạnh nhau."""
    tiles = []
    for name, mask in (("khong on dinh hinh", plain), ("co on dinh hinh", stable)):
        tile = frame.copy()
        tile[mask > 0] = (0.4 * tile[mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(tile, f"{name}: {100 * (mask > 0).mean():.1f}% foreground", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    cv2.imwrite(str(out), cv2.hconcat(tiles))


def main() -> None:
    """Đọc tham số và ghi hai ảnh minh họa."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="video cần xem độ trôi")
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/explain/on_dinh_hinh")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    drift, masks = run(args.video, config, args.frame)
    plot_drift(drift, args.out / "1_do_troi.png")
    if masks is not None:
        draw_masks(*masks, args.out / "2_mat_na.png")
    print(f"Đã ghi ảnh vào {args.out}")


if __name__ == "__main__":
    main()
