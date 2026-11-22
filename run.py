"""Chạy TOÀN BỘ hệ thống trên một cảnh hai camera, rồi chấm điểm nếu có đáp án.

Cách dùng:
    python run.py --scene data/synthetic_v2/s2 [--out outputs/he_thong/s2] [--debug-video]

Thư mục cảnh cần có cam_a/video.mp4 và cam_b/video.mp4. Homography A → B lấy từ scene.json
(video tổng hợp) hoặc homography.json (từ tools/calibrate_floor.py), nếu có.
Các bước:
1. Mỗi camera: tìm người → bám người (src/pipeline.py).
2. Mỗi camera: nối các đoạn track bị đứt của cùng một người (src/tracklet_linking.py).
3. Hai camera: nhận dạng lại, gán ID toàn cục (src/reid.py).
4. Chấm điểm nếu có đáp án (src/system_eval.py), cả trước lẫn sau khi nối tracklet.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src import homography
from src.color_features import gray_world_gains
from src.gallery import Observation
from src.metrics import detection_pr
from src.mot_metrics import mot_metrics
from src.pipeline import process_camera
from src.reid import VARIANTS, assign_global_ids, describe_tracks, read_frames, similarity_tables
from src.system_eval import evaluate_reid, map_tracks_to_gt, read_pairs
from src.tracklet_linking import link_tracklets, relabel, segments_from_rows

CAMERAS = ("cam_a", "cam_b")
TRACK_HEADER = ["frame", "track_id", "state", "x", "y", "w", "h", "n_points"]


def load_homography(scene_dir: Path) -> np.ndarray | None:
    """Homography A → B: ưu tiên homography.json (click tay), sau đó scene.json (video tổng hợp)."""
    if (scene_dir / "homography.json").exists():
        return homography.load(scene_dir / "homography.json")["H_ab"]
    if (scene_dir / "scene.json").exists():
        return np.asarray(json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))["H_ab"])
    return None


def observations_from_rows(rows, min_frames: int) -> dict[int, list[Observation]]:
    """Lần xuất hiện của từng track đã xác nhận; lúc merged thì không sạch. Bỏ track quá ngắn."""
    observations: dict[int, list[Observation]] = defaultdict(list)
    for frame, track_id, state, x, y, w, h, _ in rows:
        if state in ("confirmed", "static", "merged"):
            observations[track_id].append(Observation(frame, (x, y, w, h), state != "merged"))
    return {tid: obs for tid, obs in observations.items() if len(obs) >= min_frames}


def write_csv(path: Path, header, rows) -> None:
    """Ghi một file CSV."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def link_camera(video: str, result, config: dict) -> tuple[list[tuple], dict[int, int]]:
    """Nối tracklet của một camera; màu của mỗi đoạn lấy từ gallery (src/reid.py)."""
    gains = gray_world_gains(read_frames(video, {0})[0])
    observations = observations_from_rows(result.rows, 1)
    described = describe_tracks(video, observations, result.image_size, gains, "sift", config["features"])
    colors = {tid: d.colors for tid, d in described.items()}
    mapping = link_tracklets(segments_from_rows(result.rows), colors, config["tracklet_linking"])
    return relabel(result.rows, mapping), mapping


def main() -> None:
    """Chạy hệ thống và in báo cáo."""
    parser = argparse.ArgumentParser(description="Chạy toàn bộ hệ thống trên một cảnh hai camera")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--debug-video", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.out or Path("outputs/he_thong") / args.scene.name
    results, linked = {}, {}
    for camera in CAMERAS:
        video = str(args.scene / camera / "video.mp4")
        results[camera] = process_camera(video, camera, out_dir / camera, config, args.debug_video)
        linked[camera], mapping = link_camera(video, results[camera], config)
        write_csv(out_dir / camera / "tracks_linked.csv", TRACK_HEADER, linked[camera])
        merged = sum(old != new for old, new in mapping.items())
        print(f"{camera}: xong, nối {merged} đoạn track bị đứt")

    method = config["features"]["method"]
    described = {}
    for camera in CAMERAS:
        video = str(args.scene / camera / "video.mp4")
        gains = gray_world_gains(read_frames(video, {0})[0])
        observations = observations_from_rows(linked[camera], int(config["reid"]["min_track_frames"]))
        described[camera] = describe_tracks(video, observations, results[camera].image_size, gains, method, config["features"])
    tables = similarity_tables(described["cam_a"], described["cam_b"], load_homography(args.scene), method, config["reid"])
    ids_a, ids_b = sorted(described["cam_a"]), sorted(described["cam_b"])
    global_rows = {v: assign_global_ids(ids_a, ids_b, tables[v], float(config["reid"]["match_threshold"])) for v in VARIANTS}
    for variant, rows in global_rows.items():
        write_csv(out_dir / f"global_ids_{variant}.csv", ["cam", "track_id", "global_id"], rows)
    report(args.scene, out_dir, results, global_rows)


def report(scene_dir: Path, out_dir: Path, results, global_rows) -> None:
    """In và lưu bảng điểm khi thư mục cảnh có đáp án."""
    summary = {}
    mains = {}
    for camera, result in results.items():
        gt_csv = scene_dir / camera / "gt_tracking.csv"
        if not gt_csv.exists():
            continue
        pr = detection_pr(result.detections_csv, gt_csv)
        raw = mot_metrics(result.tracks_csv, gt_csv)
        linked_csv = out_dir / camera / "tracks_linked.csv"
        joined = mot_metrics(linked_csv, gt_csv)
        _, mains[camera] = map_tracks_to_gt(linked_csv, gt_csv)
        summary[camera] = {
            "precision": round(pr["precision"], 3), "recall": round(pr["recall"], 3),
            "truoc_khi_noi": {"MOTA": round(raw["mota"], 3), "IDF1": round(raw["idf1"], 3), "IDSW": raw["idsw"]},
            "sau_khi_noi": {"MOTA": round(joined["mota"], 3), "IDF1": round(joined["idf1"], 3), "IDSW": joined["idsw"]},
        }
    if len(mains) == 2 and (scene_dir / "gt_reid.csv").exists():
        pairs = read_pairs(scene_dir / "gt_reid.csv")
        for variant, rows in global_rows.items():
            summary[f"reid_{variant}"] = evaluate_reid(rows, mains["cam_a"], mains["cam_b"], pairs)["accuracy_str"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
