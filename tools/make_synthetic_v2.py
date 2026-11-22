"""Sinh video tổng hợp v2 (2 camera chồng vùng nhìn, 4 người) kèm đầy đủ đáp án (QĐ-16).

Cách dùng (chạy từ thư mục gốc repo, bắt buộc có -m):
    python -m tools.make_synthetic_v2 --out data/synthetic_v2
    python -m tools.make_synthetic_v2 --out data/synthetic_v2 --scenario s2 s5 --explain outputs/explain/video_tong_hop

Mỗi kịch bản ghi vào <out>/<kịch bản>/:
- cam_a/video.mp4, cam_b/video.mp4 và metadata.json của từng camera
- cam_a/gt_tracking.csv, cam_b/gt_tracking.csv: đáp án từng camera (có occluded, keyframe)
- gt_reid.csv: ai ở cam A là ai ở cam B
- scene.json: homography sàn → ảnh, homography A → B, điểm băng keo trong từng ảnh
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.common.homography import project
from tools.annotate import write_gt
from tools.synthetic_render import make_background, render_frame, visible_annotations
from tools.synthetic_scenarios import SCENARIOS
from tools.synthetic_scene import (
    CAMERAS, EMPTY_SECONDS, IMAGE_SIZE, Layout, Scenario,
    floor_to_image, n_frames, person_box,
)


def camera_geometry(layout: Layout, camera: str) -> dict:
    """Thông tin hình học của một camera, ghi vào scene.json."""
    H = floor_to_image(layout, camera)
    return {
        "floor_to_image": H.tolist(),
        "tape_marks": np.round(project(H, layout.tape_marks_m), 2).tolist(),
    }


def render_camera(scenario: Scenario, camera: str, folder: Path, fps: float, seed: int) -> list[tuple]:
    """Vẽ toàn bộ video của một camera; trả về các dòng đáp án."""
    people = scenario.people
    H = floor_to_image(scenario.layout, camera)
    rng = np.random.default_rng(seed + CAMERAS.index(camera))
    background = make_background(scenario.layout, H, rng)
    folder.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, IMAGE_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được video trong {folder}")
    rows = []
    try:
        for frame_index in range(n_frames(fps)):
            t = frame_index / fps
            visible = [(person, person.position(t)) for person in people]
            boxes = [(person, person_box(H, *pos)) for person, pos in visible if pos is not None]
            frame, labels = render_frame(background, boxes, rng)
            for gt_id, x, y, w, h, level in visible_annotations(labels, boxes):
                rows.append((frame_index, gt_id, x, y, w, h, level, 1))
            writer.write(frame)
    finally:
        writer.release()
    write_gt(folder / "gt_tracking.csv", rows)
    metadata = {"synthetic": True, "version": 2, "scenario": scenario.name, "layout": scenario.layout.name, "camera": camera, "seed": seed}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rows


def write_csv(path: Path, header: list[str], rows) -> None:
    """Ghi một file CSV nhỏ."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def generate(scenario: str, out: Path, fps: float = 30.0, seed: int = 0) -> Path:
    """Sinh một kịch bản; trả về thư mục kết quả."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Kịch bản phải là một trong {sorted(SCENARIOS)}")
    chosen = SCENARIOS[scenario]
    folder = Path(out) / scenario
    ids_by_camera = {}
    for camera in CAMERAS:
        rows = render_camera(chosen, camera, folder / camera, fps, seed)
        ids_by_camera[camera] = {row[1] for row in rows}
    both = sorted(ids_by_camera["cam_a"] & ids_by_camera["cam_b"])
    write_csv(folder / "gt_reid.csv", ["person", "cam_a_id", "cam_b_id"], [(pid, pid, pid) for pid in both])
    geometry = {camera: camera_geometry(chosen.layout, camera) for camera in CAMERAS}
    H_a = np.asarray(geometry["cam_a"]["floor_to_image"])
    H_b = np.asarray(geometry["cam_b"]["floor_to_image"])
    scene = {
        "scenario": scenario, "layout": chosen.layout.name, "description": chosen.description,
        "synthetic": True, "version": 2, "seed": seed, "fps": fps,
        "image_size": list(IMAGE_SIZE), "n_frames": n_frames(fps),
        "empty_frames": int(round(EMPTY_SECONDS * fps)),
        "H_ab": (H_b @ np.linalg.inv(H_a)).tolist(), "cameras": geometry,
    }
    (folder / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return folder


def explain(folder: Path, out_png: Path) -> None:
    """Ảnh minh họa: cam_a | cam_b ở 3 thời điểm, có box đáp án (màu theo mức che)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    total = scene["n_frames"]
    moments = [int(total * ratio) for ratio in (0.35, 0.5, 0.7)]
    colors = {0: "lime", 1: "gold", 2: "red"}
    figure, axes = plt.subplots(len(moments), 2, figsize=(12, 3.6 * len(moments)))
    for col, camera in enumerate(CAMERAS):
        with (folder / camera / "gt_tracking.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        capture = cv2.VideoCapture(str(folder / camera / "video.mp4"))
        for row_index, frame_index in enumerate(moments):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            _, frame = capture.read()
            axis = axes[row_index][col]
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for row in rows:
                if int(row["frame"]) != frame_index:
                    continue
                level = int(row["occluded"])
                box = [int(row[k]) for k in ("x", "y", "w", "h")]
                axis.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, color=colors[level], linewidth=2))
                axis.text(box[0], box[1] - 3, f"người {row['gt_id']} (che {level})", color=colors[level], fontsize=8)
            axis.set_title(f"{camera} – giây {frame_index / scene['fps']:.1f}")
            axis.axis("off")
        capture.release()
    figure.suptitle(f"Kịch bản {scene['scenario']} ({scene['description']})\nbox chỉ bao phần nhìn thấy; xanh = không che, vàng = che một phần, đỏ = che nặng")
    figure.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=90)
    plt.close(figure)


def main() -> None:
    """Sinh các kịch bản được chọn (mặc định tất cả) và, nếu có --explain, xuất ảnh minh họa."""
    parser = argparse.ArgumentParser(description="Sinh video tổng hợp v2 kèm đáp án")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", nargs="+", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explain", type=Path, help="thư mục ghi ảnh minh họa, mỗi kịch bản một ảnh")
    args = parser.parse_args()
    for scenario in args.scenario:
        folder = generate(scenario, args.out, args.fps, args.seed)
        print(f"Đã sinh {folder}")
        if args.explain:
            explain(folder, args.explain / f"{scenario}.png")


if __name__ == "__main__":
    main()
