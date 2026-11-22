"""Chạy nhận dạng lại giữa hai camera ở chế độ ORACLE (dùng khung đáp án làm track) và chấm điểm.

Cách dùng:
    python -m tools.run_reid --scene data/synthetic_v2/s4 [--method sift] [--out outputs/reid/s4]

Chế độ oracle đo riêng năng lực nhận dạng lại, không bị lỗi tìm người / bám người ảnh hưởng
(XLA 2.10). Ghi ra thư mục --out:
- reid_scores.csv: điểm giống nhau của mọi cặp (các method: keypoint, mau, san, ngoai_hinh, ngoai_hinh_san);
- global_ids_ngoai_hinh.csv và global_ids_ngoai_hinh_san.csv: kết quả gán ID toàn cục của hai biến thể.
In ra: số cặp đúng (x/y), rank-1 và bảng điểm của từng biến thể.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.reid.color_features import gray_world_gains
from src.reid.gallery import Observation
from src.evaluation.metrics import rank1, reid_metrics
from src.reid.reid import VARIANTS, assign_global_ids, describe_tracks, read_frames, similarity_tables
from tools.verify_gt import read_gt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam_a", "cam_b")


def load_observations(gt_csv: Path) -> dict[int, list[Observation]]:
    """Đọc đáp án của một camera thành các lần xuất hiện theo track; sạch nghĩa là occluded = 0."""
    observations: dict[int, list[Observation]] = defaultdict(list)
    for frame, gt_id, x, y, w, h, occluded, _ in read_gt(gt_csv):
        observations[gt_id].append(Observation(frame, (x, y, w, h), occluded == 0))
    return dict(observations)


def run(scene_dir: Path, method: str, out_dir: Path, config: dict) -> dict:
    """Chạy toàn bộ và ghi kết quả; trả về các bảng điểm và id để vẽ minh họa."""
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    image_size = scene["image_size"]
    tracks = {}
    for camera in CAMERAS:
        video = str(scene_dir / camera / "video.mp4")
        gains = gray_world_gains(read_frames(video, {0})[0])
        observations = load_observations(scene_dir / camera / "gt_tracking.csv")
        tracks[camera] = describe_tracks(video, observations, image_size, gains, method, config["features"])
    tables = similarity_tables(tracks["cam_a"], tracks["cam_b"], np.asarray(scene["H_ab"]), method, config["reid"])
    ids_a, ids_b = sorted(tracks["cam_a"]), sorted(tracks["cam_b"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "reid_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cam_a_track", "cam_b_track", "method", "score"])
        for name, table in tables.items():
            for i, id_a in enumerate(ids_a):
                for j, id_b in enumerate(ids_b):
                    if not np.isnan(table[i, j]):
                        writer.writerow([id_a, id_b, name, round(float(table[i, j]), 4)])
    for variant in VARIANTS:
        rows = assign_global_ids(ids_a, ids_b, tables[variant], float(config["reid"]["match_threshold"]))
        with (out_dir / f"global_ids_{variant}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cam", "track_id", "global_id"])
            writer.writerows(rows)
    return {"tables": tables, "ids_a": ids_a, "ids_b": ids_b, "tracks": tracks}


def print_report(scene_dir: Path, out_dir: Path, result: dict) -> None:
    """In kết quả chấm điểm của từng biến thể, kèm bảng điểm tổng."""
    gt_reid = scene_dir / "gt_reid.csv"
    for variant in VARIANTS:
        ids = reid_metrics(out_dir / f"global_ids_{variant}.csv", gt_reid)
        top1 = rank1(out_dir / "reid_scores.csv", gt_reid, variant)
        print(f"\n[{variant}] ghép đúng {ids['accuracy_str']} cặp, rank-1 {top1['accuracy_str']}")
        print("      " + "  ".join(f"B{j:<4}" for j in result["ids_b"]))
        for i, id_a in enumerate(result["ids_a"]):
            print(f"A{id_a:<4} " + "  ".join(f"{v:5.2f}" for v in result["tables"][variant][i]))


def main() -> None:
    """Đọc tham số, chạy và in báo cáo."""
    parser = argparse.ArgumentParser(description="Nhận dạng lại giữa hai camera, chế độ oracle")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--method", choices=("sift", "orb", "sift_auto", "asift"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    method = args.method or config["features"]["method"]
    out_dir = args.out or Path("outputs/reid") / f"{args.scene.name}_{method}"
    result = run(args.scene, method, out_dir, config)
    print(f"Kịch bản {args.scene.name}, phương pháp {method}; kết quả ghi ở {out_dir}")
    print_report(args.scene, out_dir, result)


if __name__ == "__main__":
    main()
