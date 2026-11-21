"""Chạy TOÀN BỘ hệ thống trên một cảnh hai camera, rồi chấm điểm nếu có đáp án.

Cách dùng:
    python run.py --scene data/synthetic_v2/hanh_lang_cua [--out outputs/he_thong/hanh_lang_cua] [--debug-video]

Thư mục cảnh cần có cam_a/video.mp4 và cam_b/video.mp4. Nếu có scene.json (video tổng hợp)
thì vạch cửa và homography lấy từ đó; nếu có homography.json (từ tools/calibrate_floor.py)
thì homography lấy từ file này.
Các bước: mỗi camera chạy tìm người → bám người → đếm cửa (src/pipeline.py); sau đó nhận
dạng lại giữa hai camera (src/reid.py) và ghép lượt qua cửa theo thời gian (src/door_matching.py).
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
from src.door_matching import match_door_events, track_links
from src.gallery import Observation
from src.metrics import detection_pr
from src.mot_metrics import mot_metrics
from src.pipeline import process_camera
from src.reid import VARIANTS, assign_global_ids, describe_tracks, read_frames, similarity_tables
from src.system_eval import evaluate_door, evaluate_reid, map_tracks_to_gt, read_pairs

CAMERAS = ("cam_a", "cam_b")


def scene_geometry(scene_dir: Path) -> tuple[dict, np.ndarray | None]:
    """Vạch cửa từng camera và homography A → B, nếu có."""
    doors, H_ab = {}, None
    scene_file = scene_dir / "scene.json"
    if scene_file.exists():
        scene = json.loads(scene_file.read_text(encoding="utf-8"))
        doors = {cam: (g["door_line"], g["door_inside_side"]) for cam, g in scene["cameras"].items()}
        H_ab = np.asarray(scene["H_ab"])
    if (scene_dir / "homography.json").exists():
        H_ab = homography.load(scene_dir / "homography.json")["H_ab"]
    return doors, H_ab


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
    doors, H_ab = scene_geometry(args.scene)
    results = {}
    for camera in CAMERAS:
        door_line, inside = doors.get(camera, (None, None))
        results[camera] = process_camera(args.scene / camera / "video.mp4", camera, out_dir / camera, config,
                                         door_line, inside, args.debug_video)
        print(f"{camera}: xong {results[camera].tracks_csv}")

    method = config["features"]["method"]
    described = {}
    for camera in CAMERAS:
        video = str(args.scene / camera / "video.mp4")
        gains = gray_world_gains(read_frames(video, {0})[0])
        observations = observations_from_rows(results[camera].rows, int(config["reid"]["min_track_frames"]))
        described[camera] = describe_tracks(video, observations, results[camera].image_size, gains, method, config["features"])
    tables = similarity_tables(described["cam_a"], described["cam_b"], H_ab, method, config["reid"])
    ids_a, ids_b = sorted(described["cam_a"]), sorted(described["cam_b"])
    global_rows = {v: assign_global_ids(ids_a, ids_b, tables[v], float(config["reid"]["match_threshold"])) for v in VARIANTS}
    for variant, rows in global_rows.items():
        write_csv(out_dir / f"global_ids_{variant}.csv", ["cam", "track_id", "global_id"], rows)
    links = match_door_events(results["cam_a"].events, results["cam_b"].events, results["cam_a"].fps,
                              float(config["reid"]["door_event_max_gap_s"]))
    write_csv(out_dir / "door_links.csv", ["cam_a_track", "cam_b_track", "gap_s"],
              [(m.event_a.track_id, m.event_b.track_id, round(m.gap_s, 3)) for m in links])
    final_ids = {(cam, track): gid for cam, track, gid in global_rows["ngoai_hinh_san"]}
    events = [(i, cam, e.frame, e.track_id, final_ids.get((cam, e.track_id), -1), e.direction)
              for cam in CAMERAS for i, e in enumerate(results[cam].events, start=1)]
    write_csv(out_dir / "events.csv", ["event_id", "camera", "frame", "track_id", "global_id", "direction"], events)
    report(args.scene, out_dir, results, global_rows, track_links(links), config)


def report(scene_dir: Path, out_dir: Path, results, global_rows, door_links, config) -> None:
    """In và lưu bảng điểm khi thư mục cảnh có đáp án."""
    summary = {}
    mains = {}
    maps = {}
    for camera, result in results.items():
        gt_csv = scene_dir / camera / "gt_tracking.csv"
        if not gt_csv.exists():
            continue
        pr = detection_pr(result.detections_csv, gt_csv)
        mot = mot_metrics(result.tracks_csv, gt_csv)
        track_to_gt, mains[camera] = map_tracks_to_gt(result.tracks_csv, gt_csv)
        maps[camera] = track_to_gt
        summary[camera] = {"precision": round(pr["precision"], 3), "recall": round(pr["recall"], 3),
                           "MOTA": round(mot["mota"], 3), "IDF1": round(mot["idf1"], 3), "IDSW": mot["idsw"]}
        if (scene_dir / "gt_events.csv").exists() and result.events:
            summary[camera]["cua"] = evaluate_door(result.events, track_to_gt, scene_dir / "gt_events.csv", result.fps,
                                                   float(config["reid"]["door_event_max_gap_s"]))
    if len(mains) == 2 and (scene_dir / "gt_reid.csv").exists():
        pairs = read_pairs(scene_dir / "gt_reid.csv")
        for variant, rows in global_rows.items():
            summary[f"reid_{variant}"] = evaluate_reid(rows, mains["cam_a"], mains["cam_b"], pairs)["accuracy_str"]
        correct = sum(maps["cam_a"].get(a) is not None and maps["cam_a"].get(a) == maps["cam_b"].get(b)
                      for a, b in door_links.items())
        summary["ghep_qua_cua"] = f"{correct}/{len(door_links)} cặp đúng"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
