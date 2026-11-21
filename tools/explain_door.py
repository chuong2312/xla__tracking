"""Ảnh minh họa cho đếm qua cửa (src/door.py).

Cách dùng:
    python -m tools.explain_door --scene data/synthetic_v2/door --out outputs/explain/dem_qua_cua

Vẽ trên cảnh trống: vạch cửa, vùng đệm hai bên vạch, đường đi của điểm chân từng
người (lấy từ đáp án), và chỗ bộ đếm phát sự kiện VÀO / RA.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.door import DoorCounter, foot_point  # noqa: E402
from tools.calibrate_floor import first_frame  # noqa: E402
from tools.verify_gt import read_gt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def band_lines(line, margin: float) -> list[np.ndarray]:
    """Hai đường song song cách vạch cửa một khoảng `margin` về hai phía (ranh giới vùng đệm)."""
    x1, y1, x2, y2 = line
    direction = np.array([x2 - x1, y2 - y1], dtype=float)
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    start, end = np.array([x1, y1]), np.array([x2, y2])
    return [np.array([start + sign * margin * normal, end + sign * margin * normal]) for sign in (1, -1)]


def main() -> None:
    """Chạy bộ đếm trên đáp án của kịch bản cửa và vẽ kết quả."""
    parser = argparse.ArgumentParser(description="Minh họa đếm qua cửa")
    parser.add_argument("--scene", type=Path, default=Path("data/synthetic_v2/door"))
    parser.add_argument("--camera", default="cam_a")
    parser.add_argument("--out", type=Path, default=Path("outputs/explain/dem_qua_cua"))
    args = parser.parse_args()
    events_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))["events"]
    geometry = json.loads((args.scene / "scene.json").read_text(encoding="utf-8"))["cameras"][args.camera]
    line = geometry["door_line"]
    counter = DoorCounter(line, events_cfg["crossing_margin"], geometry["door_inside_side"])
    paths: dict[int, list] = defaultdict(list)
    feet_by_frame: dict[tuple[int, int], tuple[float, float]] = {}
    for frame, gt_id, x, y, w, h, _, _ in sorted(read_gt(args.scene / args.camera / "gt_tracking.csv")):
        foot = foot_point((x, y, w, h))
        paths[gt_id].append(foot)
        feet_by_frame[(gt_id, frame)] = foot
        counter.update(frame, gt_id, foot)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    image = first_frame(str(args.scene / args.camera / "video.mp4"), 0, None)
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.plot([line[0], line[2]], [line[1], line[3]], color="saddlebrown", linewidth=3, label="vạch cửa")
    for index, band in enumerate(band_lines(line, events_cfg["crossing_margin"])):
        axis.plot(band[:, 0], band[:, 1], color="black", linestyle=":", linewidth=1, label="ranh giới vùng đệm" if index == 0 else None)
    for gt_id, points in paths.items():
        points = np.array(points)
        axis.plot(points[:, 0], points[:, 1], linewidth=1.5, label=f"điểm chân người {gt_id}")
    lines = []
    for number, event in enumerate(counter.events, start=1):
        x, y = feet_by_frame[(event.track_id, event.frame)]
        color = "green" if event.direction == "IN" else "red"
        word = "VÀO" if event.direction == "IN" else "RA"
        axis.scatter([x], [y], s=160, marker="o", color=color, edgecolors="white", zorder=5)
        axis.text(x, y, str(number), color="white", fontsize=8, ha="center", va="center", weight="bold", zorder=6)
        lines.append(f"{number}. người {event.track_id} {word} ở frame {event.frame}")
    axis.text(0.99, 0.02, "\n".join(lines), transform=axis.transAxes, ha="right", va="bottom",
              fontsize=9, bbox={"facecolor": "white", "alpha": 0.85})
    counts = counter.counts()
    axis.set_title(f"{args.scene.name}/{args.camera}: đếm được VÀO {counts['IN']}, RA {counts['OUT']} "
                   f"(xanh = vào, đỏ = ra; vùng đệm ±{events_cfg['crossing_margin']} px)")
    axis.legend(loc="lower left", fontsize=8)
    axis.axis("off")
    figure.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / f"{args.scene.name}_{args.camera}.png"
    figure.savefig(out_png, dpi=90)
    plt.close(figure)
    print(f"Đã ghi {out_png}; kết quả {counts}")


if __name__ == "__main__":
    main()
