"""Ảnh minh họa cho ghép cặp Hungarian (src/assignment.py).

Cách dùng:
    python -m tools.explain_assignment --scene data/synthetic_v2/s2 --camera cam_b --frame 174 --out outputs/explain/ghep_cap

Giả lập một bước của tracker: "box dự đoán" lấy từ đáp án của vài frame trước
(như thể đoán vị trí cũ), "box phát hiện" lấy từ đáp án của frame hiện tại.
Ảnh bên trái vẽ hai loại box và các cặp được ghép; ảnh bên phải là bảng IoU,
ô có viền là cặp Hungarian chọn, ô ghi "cấm" là cặp có IoU dưới ngưỡng.
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
from matplotlib.patches import Rectangle  # noqa: E402

from src.assignment import assign, iou_cost  # noqa: E402
from tools.calibrate_floor import first_frame  # noqa: E402
from tools.verify_gt import read_gt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def boxes_at(rows, frame: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Các (gt_id, box) của một frame."""
    return [(row[1], tuple(row[2:6])) for row in rows if row[0] == frame]


def center(box) -> tuple[float, float]:
    """Tâm của box (x, y, w, h)."""
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def main() -> None:
    """Vẽ ảnh minh họa ghép cặp cho một frame."""
    parser = argparse.ArgumentParser(description="Minh họa ghép cặp Hungarian")
    parser.add_argument("--scene", type=Path, default=Path("data/synthetic_v2/s2"))
    parser.add_argument("--camera", default="cam_b")
    parser.add_argument("--frame", type=int, default=174)
    parser.add_argument("--lag", type=int, default=4, help="box dự đoán lấy từ đáp án cách bao nhiêu frame trước")
    parser.add_argument("--out", type=Path, default=Path("outputs/explain/ghep_cap"))
    args = parser.parse_args()
    min_iou = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))["tracker"]["iou_gate"]
    rows = read_gt(args.scene / args.camera / "gt_tracking.csv")
    predicted = boxes_at(rows, args.frame - args.lag)
    detected = [box for _, box in boxes_at(rows, args.frame)][::-1]
    cost = iou_cost([box for _, box in predicted], detected, min_iou)
    pairs = assign(cost, max_cost=1.0 - min_iou)
    overlaps = np.where(np.isfinite(cost), 1.0 - cost, 0.0)

    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    image = first_frame(str(args.scene / args.camera / "video.mp4"), args.frame, None)
    left.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for row_index, (gt_id, box) in enumerate(predicted):
        left.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, color="deepskyblue", linestyle="--", linewidth=2))
        left.text(box[0], box[1] - 4, f"track {row_index}", color="deepskyblue", fontsize=9)
    for col_index, box in enumerate(detected):
        left.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, color="orange", linewidth=2))
        left.text(box[0], box[1] + box[3] + 12, f"phát hiện {col_index}", color="orange", fontsize=9)
    for i, j in pairs:
        (x1, y1), (x2, y2) = center(predicted[i][1]), center(detected[j])
        left.plot([x1, x2], [y1, y2], color="lime", linewidth=2)
    left.set_title(f"Frame {args.frame}: nét đứt = box dự đoán, nét liền = box phát hiện,\nxanh lá = cặp được ghép", fontsize=10)
    left.axis("off")

    right.imshow(overlaps, cmap="Greens", vmin=0, vmax=1)
    for i in range(overlaps.shape[0]):
        for j in range(overlaps.shape[1]):
            text = f"{overlaps[i, j]:.2f}" if np.isfinite(cost[i, j]) else "cấm"
            right.text(j, i, text, ha="center", va="center", fontsize=10)
    for i, j in pairs:
        right.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, color="red", linewidth=3))
    right.set_xticks(range(len(detected)), [f"phát hiện {j}" for j in range(len(detected))], rotation=30)
    right.set_yticks(range(len(predicted)), [f"track {i}" for i in range(len(predicted))])
    right.set_title(f"Bảng IoU: ô viền đỏ = cặp Hungarian chọn,\nIoU < {min_iou} bị cấm", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / f"{args.scene.name}_{args.camera}_frame{args.frame}.png"
    figure.savefig(out_png, dpi=90)
    plt.close(figure)
    print(f"Đã ghi {out_png}; các cặp (track, phát hiện): {pairs}")


if __name__ == "__main__":
    main()
