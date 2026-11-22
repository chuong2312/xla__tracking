"""Kiểm thử sự kiện diagnostics của prediction, association và recovery."""
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest

from legacy.tracker import Tracker
from tools.trace_tracking import collect_provenance_files, sha256_file


def _cfg() -> dict[str, dict[str, object]]:
    return {"tracker": {
        "winSize": [21, 21], "maxLevel": 2, "criteria": [3, 30, 0.01],
        "use_fb_check": True, "fb_thresh": 1.0, "min_points": 3,
        "max_corners": 80, "quality_level": 0.01, "min_distance": 3.0,
        "block_size": 5, "refresh_every": 3, "matcher": "hungarian",
        "iou_gate": 0.05, "confirm_hits": 1, "max_misses": 4,
        "box_update_alpha": 0.0, "blob_iou": 0.2,
        "blob_track_coverage": 0.75, "blob_center_separation": 0.35,
        "blob_grace_frames": 35, "hist_bins": [16, 16], "hist_corr_min": 0.7,
    }}


def _frame(boxes, colors, seed=7):
    frame = np.full((80, 80, 3), (20, 20, 20), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for (x, y, w, h), color in zip(boxes, colors):
        texture = rng.integers(0, 20, (h, w, 1), dtype=np.uint8)
        patch = np.clip(np.asarray(color, dtype=np.uint8).reshape(1, 1, 3) + texture, 0, 255)
        frame[y:y + h, x:x + w] = patch
    return frame


def test_update_logs_prediction_hungarian_and_recovery_evidence(caplog):
    boxes = [(10, 20, 20, 24), (30, 20, 20, 24)]
    colors = [(0, 0, 230), (0, 220, 0)]
    clean = _frame(boxes, colors)
    tracker = Tracker(_cfg())
    tracker.update(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), boxes, frame_bgr=clean)
    merged_box = (10, 20, 40, 24)
    merged = _frame([merged_box], [(120, 120, 120)])
    tracker.update(cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY), [merged_box], frame_bgr=merged)

    with caplog.at_level(logging.INFO, logger="legacy.tracker"):
        tracker.update(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), boxes, frame_bgr=clean)

    events = [json.loads(record.getMessage()) for record in caplog.records if record.name == "legacy.tracker"]
    prediction = next(event for event in events if event["event"] == "prediction")
    assignment = next(event for event in events if event["event"] == "association_matrix")
    recovery = next(event for event in events if event["event"] == "recovery_matrix")
    assert {"points_before_lk", "points_after_status_filter", "points_after_fb_check",
            "points_used_for_median", "median_dx", "median_dy", "prediction_applied",
            "prediction_fallback_reason"} <= prediction.keys()
    assert {"track_boxes", "detection_boxes", "iou_matrix", "cost_matrix",
            "candidate_eligible", "assignments"} <= assignment.keys()
    assert {"candidates", "hungarian_assignments", "grace_remaining"} <= recovery.keys()
    assert recovery["candidates"] and all(
        {"track_id", "detection_index", "iou", "hist_corr", "geometry_pass",
         "appearance_pass", "candidate_eligible", "assigned", "rejection_reason"} <= candidate.keys()
        for candidate in recovery["candidates"]
    )


def test_run_manifest_hashes_untracked_tests_and_synthetic_inputs():
    """Manifest enumerates working-tree source/tests and actual synthetic inputs."""
    root = Path(__file__).resolve().parents[2]
    video = root / "data/synthetic/cross/cam_a/video.mp4"
    gt = root / "data/synthetic/cross/gt_tracking.csv"
    config = root / "config.yaml"
    records = collect_provenance_files(root, config, video, gt)
    paths = {record["relative_path"] for record in records}

    assert "legacy/tracker.py" in paths
    assert "tests/legacy/test_tracker_blob.py" in paths
    assert "tests/legacy/test_tracker_diagnostics.py" in paths
    assert "config.yaml" in paths
    assert "tools/make_synthetic.py" in paths
    assert "data/synthetic/cross/cam_a/video.mp4" in paths
    assert "data/synthetic/cross/gt_tracking.csv" in paths
    assert all(len(record["sha256"]) == 64 for record in records)


def test_sha256_file_reads_exact_file_bytes(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"synthetic-input\x00bytes")
    assert sha256_file(path) == "5d8072098110453c41955a7783947f0f6be7843b9117a9fbfa808414b1ae62a1"


def test_stale_track_false_positive_assignment_is_visible_in_trace(caplog):
    """Ghi nhận stale-track/false-positive cạnh tranh mà không áp policy mới."""
    old_box = (10, 20, 20, 24)
    new_box = (50, 20, 20, 24)
    first = _frame([old_box], [(0, 0, 230)])
    tracker = Tracker(_cfg())
    tracker.update(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), [old_box], frame_bgr=first)

    moved = _frame([new_box], [(0, 0, 230)])
    tracker.update(cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY), [], frame_bgr=moved)
    with caplog.at_level(logging.INFO, logger="legacy.tracker"):
        result = tracker.update(
            cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY), [old_box, new_box], frame_bgr=moved,
        )

    events = [json.loads(record.getMessage()) for record in caplog.records if record.name == "legacy.tracker"]
    matrix = next(event for event in events if event["event"] == "association_matrix")
    assignment = next(event for event in events if event["event"] == "normal_assignment_result")
    assert matrix["track_boxes"] == [list(old_box)]
    assert matrix["detection_boxes"] == [list(old_box), list(new_box)]
    assert matrix["assignments"] == [[0, 0]]
    assert assignment["assignments"] == [{"track_id": 1, "detection_index": 0}]
    assert {track.id for track in result} == {1, 2}
