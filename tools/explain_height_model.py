"""Ảnh minh họa cho mô hình chiều cao người và việc gộp mảnh (src/height_model.py, QĐ-26).

Cách dùng:
    python -m tools.explain_height_model --video data/real/4p-c1.mp4 --frame 1800 --out outputs/explain/chieu_cao

Ghi ra hai ảnh:
1. `1_mo_hinh_chieu_cao.png`: mỗi chấm là một box tìm được (vị trí chân, chiều cao). Chấm đỏ
   là phân vị 90% của từng khoảng, đường xanh là chiều cao mong đợi h(y) = a·y + b. Đám chấm
   thấp bên dưới đường là các mảnh người (chỉ đầu, chỉ chân).
2. `2_gop_manh.png`: cùng một frame, trái là box KHÔNG gộp mảnh, phải là box CÓ gộp mảnh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.detection import PersonDetector  # noqa: E402
from src.pipeline import estimate_heights  # noqa: E402
from src.utils import iter_video  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect_boxes(video: Path, config: dict, frames: int) -> list:
    """Box tìm được (không gộp mảnh) trên `frames` frame đầu."""
    detector = PersonDetector(config["detection"])
    boxes = []
    for index, frame in iter_video(video):
        if index >= frames:
            break
        boxes.extend(detector.detect(frame))
        detector.update_background(frame, [])
    return boxes


def plot_model(boxes, model, out: Path) -> None:
    """Đám chấm (y chân, chiều cao), các điểm phân vị và đường h(y)."""
    feet = [y + h for _, y, _, h in boxes]
    heights = [h for *_, h in boxes]
    plt.figure(figsize=(8, 5))
    plt.scatter(feet, heights, s=2, alpha=0.2, color="gray", label="box tìm được (chưa gộp mảnh)")
    ys, hs = zip(*model.points)
    plt.scatter(ys, hs, color="red", zorder=3, label="phân vị 90% từng khoảng")
    line_y = [min(feet), max(feet)]
    plt.plot(line_y, [model.expected(y) for y in line_y], color="blue", label=f"h(y) = {model.a:.2f}·y + {model.b:.1f}")
    plt.xlabel("vị trí chân trong ảnh, y (pixel, càng lớn càng gần camera)")
    plt.ylabel("chiều cao box (pixel)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def draw_frame(video: Path, config: dict, model, frame_index: int, out: Path) -> None:
    """Hai detector chạy song song đến `frame_index`; vẽ box của cả hai cạnh nhau."""
    plain = PersonDetector(config["detection"])
    grouped = PersonDetector(config["detection"], heights=model)
    for index, frame in iter_video(video):
        results = [(name, det.detect(frame)) for name, det in (("khong gop", plain), ("co gop", grouped))]
        for det in (plain, grouped):
            det.update_background(frame, [])
        if index == frame_index:
            tiles = []
            for name, boxes in results:
                tile = frame.copy()
                for x, y, w, h in boxes:
                    cv2.rectangle(tile, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(tile, f"{name}: {len(boxes)} box", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                tiles.append(tile)
            image = cv2.hconcat(tiles)
            cv2.imwrite(str(out), cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            return


def main() -> None:
    """Đọc tham số và ghi hai ảnh minh họa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "data/real/4p-c1.mp4")
    parser.add_argument("--frame", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/explain/chieu_cao")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    model = estimate_heights(args.video, config)
    if model is None:
        raise SystemExit("Không ước lượng được mô hình chiều cao: quá ít box.")
    boxes = collect_boxes(args.video, config, int(config["detection"]["height_model"]["frames"]))
    plot_model(boxes, model, args.out / "1_mo_hinh_chieu_cao.png")
    draw_frame(args.video, config, model, args.frame, args.out / "2_gop_manh.png")
    print(f"Đã ghi ảnh vào {args.out}")


if __name__ == "__main__":
    main()
