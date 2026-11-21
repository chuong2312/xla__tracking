"""Ảnh minh họa cho ghép sự kiện qua cửa giữa hai camera (src/door_matching.py).

Cách dùng:
    python -m tools.explain_door_matching --scene data/synthetic_v2/hanh_lang_cua --out outputs/explain/ghep_su_kien_cua

Đếm cửa ở từng camera, dùng đáp án làm track. Track ở camera B được cố ý đánh số
khác (cộng thêm 100), vì ngoài đời mỗi camera tự đặt số riêng. Sau đó vẽ dòng thời gian:
hàng trên là camera A, hàng dưới là camera B, đường nối là cặp sự kiện được ghép.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.door import DoorCounter, foot_point  # noqa: E402
from src.door_matching import match_door_events  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROWS = {"cam_a": 1.0, "cam_b": 0.0}
ID_OFFSET = {"cam_a": 0, "cam_b": 100}


def count_events(scene_dir: Path, camera: str, geometry: dict, margin: float) -> list:
    """Chạy bộ đếm cửa trên đáp án của một camera; số track của camera B được cộng 100."""
    counter = DoorCounter(geometry["door_line"], margin, geometry["door_inside_side"])
    with (scene_dir / camera / "gt_tracking.csv").open(encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: (int(r["frame"]), int(r["gt_id"])))
    for row in rows:
        box = [int(row[k]) for k in ("x", "y", "w", "h")]
        counter.update(int(row["frame"]), int(row["gt_id"]) + ID_OFFSET[camera], foot_point(box))
    return counter.events


def main() -> None:
    """Vẽ dòng thời gian các sự kiện qua cửa và các cặp được ghép."""
    parser = argparse.ArgumentParser(description="Minh họa ghép sự kiện qua cửa giữa hai camera")
    parser.add_argument("--scene", type=Path, default=Path("data/synthetic_v2/hanh_lang_cua"))
    parser.add_argument("--out", type=Path, default=Path("outputs/explain/ghep_su_kien_cua"))
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    scene = json.loads((args.scene / "scene.json").read_text(encoding="utf-8"))
    fps = scene["fps"]
    events = {
        camera: count_events(args.scene, camera, scene["cameras"][camera], config["events"]["crossing_margin"])
        for camera in CAMERA_ROWS
    }
    max_gap = config["reid"]["door_event_max_gap_s"]
    matches = match_door_events(events["cam_a"], events["cam_b"], fps, max_gap)

    figure, axis = plt.subplots(figsize=(11, 4))
    for camera, row in CAMERA_ROWS.items():
        for event in events[camera]:
            color = "green" if event.direction == "IN" else "red"
            word = "VÀO" if event.direction == "IN" else "RA"
            t = event.frame / fps
            axis.scatter([t], [row], s=140, color=color, zorder=3)
            axis.annotate(f"track {event.track_id}\n{word}", (t, row), textcoords="offset points",
                          xytext=(0, 12 if row else -30), ha="center", fontsize=9)
    for match in matches:
        t_a, t_b = match.event_a.frame / fps, match.event_b.frame / fps
        axis.plot([t_a, t_b], [1.0, 0.0], color="blue", linewidth=2, zorder=2)
        axis.text((t_a + t_b) / 2 + 0.05, 0.5, f"lệch {match.gap_s:.2f} s", color="blue", fontsize=9)
    axis.set_yticks([0.0, 1.0], ["camera B", "camera A"])
    axis.set_ylim(-0.6, 1.6)
    axis.set_xlabel("thời gian (giây)")
    axis.set_title(f"Ghép lượt qua cửa giữa hai camera: cùng chiều và lệch ≤ {max_gap} s "
                   f"(xanh lá = vào, đỏ = ra, đường xanh dương = cặp được ghép)", fontsize=10)
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / f"{args.scene.name}.png"
    figure.savefig(out_png, dpi=90)
    plt.close(figure)
    links = {m.event_a.track_id: m.event_b.track_id for m in matches}
    print(f"Đã ghi {out_png}; track A → track B: {links}")


if __name__ == "__main__":
    main()
