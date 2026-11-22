"""Acceptance tests cho ba quỹ đạo giao cắt với detections oracle."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from legacy.config import load_config
from src.evaluation.metrics import tracking_metrics
from legacy.tracker import Tracker
from src.common.utils import iou


def _crossing_boxes(frame_index: int) -> dict[int, tuple[int, int, int, int]]:
    """Tạo ba quỹ đạo hội tụ, giao cắt rồi tiếp tục tách nhau."""
    moving_frames = 14
    t = min(frame_index, moving_frames - 1)
    return {
        0: (14 + 8 * t, 68, 12, 20),
        1: (68, 10 + 8 * t, 12, 20),
        2: (14 + 8 * t, 10 + 8 * t, 12, 20),
    }


def _render_frame(
    boxes: dict[int, tuple[int, int, int, int]],
    textures: dict[int, np.ndarray],
) -> np.ndarray:
    """Vẽ texture grayscale; thứ tự compositing cố định và không là tín hiệu ID."""
    frame = np.full((160, 160), 18, dtype=np.uint8)
    for gt_id in (0, 1, 2):
        x, y, w, h = boxes[gt_id]
        frame[y:y + h, x:x + w] = textures[gt_id]
    return frame


def test_three_real_crossing_trajectories_have_zero_switch_and_full_coverage(tmp_path):
    """Ba GT phải giao cắt thật, confirmed trước scoring và đạt coverage config."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "config.yaml")
    raw_cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    tracker = Tracker(cfg)
    warmup_frames = int(cfg.tracker["confirm_hits"])
    scoring_start = warmup_frames
    frame_count = 40
    rng = np.random.default_rng(2026)
    textures = {
        gt_id: rng.integers(20, 236, (20, 12), dtype=np.uint8)
        for gt_id in (0, 1, 2)
    }
    expected_ids: dict[int, int] = {}
    track_rows: list[list[object]] = []
    gt_rows: list[list[object]] = []
    gt_boxes_by_frame: dict[int, dict[int, tuple[int, int, int, int]]] = {}

    for frame_index in range(frame_count):
        boxes = _crossing_boxes(frame_index)
        gt_boxes_by_frame[frame_index] = boxes
        gray = _render_frame(boxes, textures)
        detections = [boxes[gt_id] for gt_id in (0, 1, 2)]
        returned = tracker.update(gray, detections)
        if frame_index == 0:
            expected_ids = {gt_id: gt_id + 1 for gt_id in (0, 1, 2)}
        if frame_index == scoring_start - 1:
            for gt_id, box in boxes.items():
                candidate = next((t for t in returned if t.id == expected_ids[gt_id]), None)
                assert candidate is not None and candidate.state == "confirmed", (
                    f"track GT {gt_id} chưa confirmed trước scoring start={scoring_start}"
                )
                assert iou(candidate.box, box) >= float(raw_cfg["metrics"]["iou_threshold"]), (
                    f"identity mapping sai trước scoring start cho GT {gt_id}"
                )
        if frame_index >= scoring_start:
            for gt_id, box in boxes.items():
                gt_rows.append([frame_index, gt_id, *box])
            for track in returned:
                track_rows.append([frame_index, track.id, track.state, *track.box, len(track.points)])

    track_csv = tmp_path / "crossing_tracks.csv"
    gt_csv = tmp_path / "crossing_gt.csv"
    with track_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "track_id", "state", "x", "y", "w", "h", "n_points"])
        writer.writerows(track_rows)
    with gt_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "gt_id", "x", "y", "w", "h"])
        writer.writerows(gt_rows)

    metrics = tracking_metrics(track_csv, gt_csv)
    threshold = float(raw_cfg["metrics"]["iou_threshold"])
    per_gt_matched = {gt_id: 0 for gt_id in (0, 1, 2)}
    per_gt_total = {gt_id: 0 for gt_id in (0, 1, 2)}
    tracks_by_frame: dict[int, list[list[object]]] = {}
    for row in track_rows:
        if row[2] == "confirmed":
            tracks_by_frame.setdefault(int(row[0]), []).append(row)
    for frame_index in range(scoring_start, frame_count):
        boxes = gt_boxes_by_frame[frame_index]
        gt_ids = list(boxes)
        observations = tracks_by_frame.get(frame_index, [])
        overlaps = np.asarray([
            [iou(boxes[gt_id], tuple(map(int, row[3:7]))) for row in observations]
            for gt_id in gt_ids
        ], dtype=float).reshape((len(gt_ids), len(observations)))
        if overlaps.size:
            rows, cols = linear_sum_assignment(1.0 - overlaps)
            for left, right in zip(rows, cols):
                if overlaps[left, right] >= threshold:
                    per_gt_matched[gt_ids[int(left)]] += 1
        for gt_id in gt_ids:
            per_gt_total[gt_id] += 1
    coverage = {gt_id: per_gt_matched[gt_id] / per_gt_total[gt_id] for gt_id in per_gt_total}
    ratio = float(raw_cfg["metrics"]["mostly_tracked_ratio"])

    assert metrics["n_gt_ids"] == 3
    assert metrics["mostly_tracked"] == 3
    assert all(value >= ratio for value in coverage.values()), f"coverage per GT: {coverage}, required={ratio}"
    assert metrics["id_switches"] == 0, f"tracking metrics={metrics}, coverage={coverage}"
