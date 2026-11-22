"""Chạy TOÀN BỘ hệ thống trên một cảnh hai camera, rồi chấm điểm nếu có đáp án.

Cách dùng:
    python run.py --scene data/canh/epfl_lab4p_c0_c1 [--out outputs/he_thong/epfl_lab4p_c0_c1] [--debug-video] [--max-frames N]

Thư mục cảnh cần có cam_a/video.<đuôi> và cam_b/video.<đuôi> (mp4, avi, mov…; xem data/README.md).
Nếu chỉ có cam_a thì chạy tìm người, bám người, nối tracklet cho một camera và bỏ qua nhận dạng lại.
Homography A → B lấy từ homography.json (tools/calibrate_floor.py hoặc tools/epfl_homography.py), nếu có.
Nếu camera có file đáp án gt_tracking.csv (và cảnh có gt_reid.csv) thì chấm điểm.
Các bước:
1. Mỗi camera: tìm người → bám người (src/pipeline.py).
2. Mỗi camera: nối các đoạn track bị đứt của cùng một người (src/tracking/tracklet_linking.py).
3. Hai camera: nhận dạng lại, gán ID toàn cục (src/reid/reid.py).
4. Chấm điểm nếu có đáp án (src/evaluation/system_eval.py), cả trước lẫn sau khi nối tracklet.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.common import homography
from src.common.utils import camera_video
from src.reid.color_features import gray_world_gains
from src.reid.gallery import Observation
from src.evaluation.metrics import detection_pr
from src.evaluation.mot_metrics import mot_metrics
from src.pipeline import process_camera
from src.reid.reid import VARIANTS, assign_global_ids, describe_tracks, read_frames, similarity_tables
from src.evaluation.system_eval import evaluate_reid, map_tracks_to_gt, read_pairs
from src.tracking.tracklet_linking import link_tracklets, relabel, segments_from_rows

CAMERAS = ("cam_a", "cam_b")
TRACK_HEADER = ["frame", "track_id", "state", "x", "y", "w", "h", "n_points"]


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
    """Nối tracklet của một camera; màu của mỗi đoạn lấy từ gallery (src/reid/reid.py)."""
    gains = gray_world_gains(read_frames(video, {0})[0])
    observations = observations_from_rows(result.rows, 1)
    described = describe_tracks(video, observations, result.image_size, gains, "sift", config["features"])
    colors = {tid: d.colors for tid, d in described.items()}
    mapping = link_tracklets(segments_from_rows(result.rows), colors, config["tracklet_linking"])
    return relabel(result.rows, mapping), mapping


def main() -> None:
    """Chạy hệ thống và in báo cáo."""
    parser = argparse.ArgumentParser(description="Chạy toàn bộ hệ thống trên một cảnh hai camera (hoặc chỉ cam_a)")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--debug-video", action="store_true")
    parser.add_argument("--max-frames", type=int, help="chỉ xử lý bấy nhiêu frame đầu (chạy thử nhanh)")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.out or Path("outputs/he_thong") / args.scene.name
    cameras = [camera for camera in CAMERAS if camera_video(args.scene / camera) is not None]
    if not cameras:
        raise SystemExit(f"Không thấy video trong {args.scene}/cam_a/ (cần file video.mp4, video.avi, video.mov…; xem data/README.md)")
    results, linked = {}, {}
    for camera in cameras:
        video = str(camera_video(args.scene / camera))
        results[camera] = process_camera(video, camera, out_dir / camera, config, args.debug_video, args.max_frames)
        linked[camera], mapping = link_camera(video, results[camera], config)
        write_csv(out_dir / camera / "tracks_linked.csv", TRACK_HEADER, linked[camera])
        merged = sum(old != new for old, new in mapping.items())
        print(f"{camera}: xong, nối {merged} đoạn track bị đứt")
    if len(cameras) < 2:
        print("Chỉ có một camera: bỏ qua bước nhận dạng lại giữa hai camera.")
        report(args.scene, out_dir, results, {})
        return
    global_rows = reidentify(args.scene, out_dir, results, linked, config)
    report(args.scene, out_dir, results, global_rows)


def reidentify(scene: Path, out_dir: Path, results, linked, config: dict) -> dict:
    """Bước F, G: mô tả ngoại hình từng track rồi gán ID toàn cục giữa cam_a và cam_b."""
    method = config["features"]["method"]
    described = {}
    for camera in CAMERAS:
        video = str(camera_video(scene / camera))
        gains = gray_world_gains(read_frames(video, {0})[0])
        observations = observations_from_rows(linked[camera], int(config["reid"]["min_track_frames"]))
        described[camera] = describe_tracks(video, observations, results[camera].image_size, gains, method, config["features"])
    tables = similarity_tables(described["cam_a"], described["cam_b"], homography.scene_homography(scene), method, config["reid"])
    ids_a, ids_b = sorted(described["cam_a"]), sorted(described["cam_b"])
    global_rows = {v: assign_global_ids(ids_a, ids_b, tables[v], float(config["reid"]["match_threshold"])) for v in VARIANTS}
    for variant, rows in global_rows.items():
        write_csv(out_dir / f"global_ids_{variant}.csv", ["cam", "track_id", "global_id"], rows)
    return global_rows


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
