"""Kiểm thử hệ thống mới: nền, tìm người, phân loại blob, tracker mới, chạy trọn một cảnh."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.background import MedianRunningBackground
from src.blob_analysis import find_merged, join_split_detections
from src.detection import PersonDetector
from src.people_tracker import PeopleTracker
from src.pipeline import process_camera
from src.mot_metrics import mot_metrics
from tools.make_synthetic_v2 import generate

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _background(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(120 + rng.normal(0, 2, (120, 160, 3)), 0, 255).astype(np.uint8)


def _textured(frame, box, seed):
    x, y, w, h = box
    rng = np.random.default_rng(seed)
    frame[y:y + h, x:x + w] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return frame


def test_median_background_finds_new_object_and_ignores_static_scene():
    model = MedianRunningBackground(dict(CONFIG["detection"], init_frames=5))
    for _ in range(5):
        assert not model.apply(_background()).any()
    frame = _textured(_background(), (40, 30, 20, 50), seed=1)
    mask = model.apply(frame)
    assert mask[55, 50] == 255 and mask[5, 5] == 0


def test_selective_update_keeps_standing_person_out_of_background():
    cfg = dict(CONFIG["detection"], init_frames=3, update_rate=0.5)
    model = MedianRunningBackground(cfg)
    for _ in range(3):
        model.apply(_background())
    person = _textured(_background(), (40, 30, 20, 50), seed=2)
    protect = np.zeros(person.shape[:2], np.uint8)
    protect[30:80, 40:60] = 255
    for _ in range(20):
        model.update(person, protect)
    assert model.apply(person)[55, 50] == 255


def test_detector_returns_person_sized_box():
    detector = PersonDetector(dict(CONFIG["detection"], init_frames=3, min_area=50))
    for _ in range(3):
        detector.detect(_background())
    boxes = detector.detect(_textured(_background(), (40, 30, 20, 50), seed=3))
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert abs(x - 40) <= 3 and abs(h - 50) <= 6


def test_split_pieces_are_joined_only_when_stacked_vertically():
    predicted = {1: (10, 10, 30, 80)}
    stacked = join_split_detections([(12, 10, 26, 35), (14, 50, 22, 38)], predicted, 0.7)
    side_by_side = join_split_detections([(10, 10, 14, 80), (26, 10, 14, 80)], predicted, 0.7)
    assert stacked == [(12, 10, 26, 78)]
    assert len(side_by_side) == 2


def test_blob_containing_two_tracks_is_merged():
    predicted = {1: (10, 10, 20, 60), 2: (25, 12, 20, 60)}
    assert find_merged([(8, 8, 40, 66)], predicted, 0.6) == {0: {1, 2}}


def _run_tracker(frames_and_boxes):
    tracker = PeopleTracker(CONFIG["people_tracker"], CONFIG["kalman"], CONFIG["features"])
    history = []
    for frame, boxes in frames_and_boxes:
        history.append({t.id: t.state for t in tracker.update(frame, boxes, None)})
    return history


def test_person_who_stops_and_loses_detection_becomes_static_not_lost():
    sequence = []
    for step in range(40):
        x = 20 + min(step, 15) * 3
        frame = _textured(_background(), (x, 30, 20, 50), seed=4)
        boxes = [(x, 30, 20, 50)] if step < 25 else []
        sequence.append((frame, boxes))
    history = _run_tracker(sequence)
    assert history[-1].get(1) == "static"


def test_two_people_crossing_keep_their_ids():
    sequence = []
    for step in range(40):
        a = (10 + 3 * step, 30, 18, 50)
        b = (130 - 3 * step, 34, 18, 50)
        frame = _textured(_textured(_background(), a, 5), b, 6)
        if abs(a[0] - b[0]) < 20:
            blob = (min(a[0], b[0]), 30, max(a[0], b[0]) + 18 - min(a[0], b[0]), 54)
            boxes = [blob]
        else:
            boxes = [a, b]
        sequence.append((frame, boxes))
    tracker = PeopleTracker(CONFIG["people_tracker"], CONFIG["kalman"], CONFIG["features"])
    positions = {}
    for frame, boxes in sequence:
        tracks = tracker.update(frame, boxes, None)
    for track in tracks:
        positions[track.id] = track.box[0]
    assert set(positions) == {1, 2}
    assert positions[1] > positions[2]


@pytest.fixture(scope="module")
def s1_scene(tmp_path_factory):
    return generate("s1", tmp_path_factory.mktemp("s1"), fps=10.0, seed=0)


def test_pipeline_on_synthetic_scene_tracks_everyone_without_switches(s1_scene, tmp_path):
    cfg = dict(CONFIG, detection=dict(CONFIG["detection"], init_frames=10))
    result = process_camera(s1_scene / "cam_a" / "video.mp4", "cam_a", tmp_path, cfg)
    scores = mot_metrics(result.tracks_csv, s1_scene / "cam_a" / "gt_tracking.csv", 0.5)
    assert scores["idsw"] == 0
    assert scores["mota"] > 0.7


def test_run_py_writes_summary(s1_scene, tmp_path):
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "run.py"), "--scene", str(s1_scene), "--out", str(out)],
                   check=True, cwd=ROOT, capture_output=True)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {"cam_a", "cam_b", "reid_ngoai_hinh", "reid_ngoai_hinh_san"} <= set(summary)
    assert "sau_khi_noi" in summary["cam_a"]
    for name in ("cam_a/tracks_linked.csv", "global_ids_ngoai_hinh.csv", "global_ids_ngoai_hinh_san.csv"):
        assert (out / name).exists()


def test_track_already_merged_stays_in_blob_with_lower_coverage():
    """Ngưỡng có độ trễ: track đang gộp chỉ cần 30% box nằm trong blob là vẫn tính trong blob."""
    predicted = {1: (10, 10, 20, 60), 2: (36, 10, 20, 60)}
    blob = [(8, 8, 36, 66)]
    assert find_merged(blob, predicted, 0.6) == {}
    assert find_merged(blob, predicted, 0.6, sticky_ids={2}, sticky_coverage=0.3) == {0: {1, 2}}


def test_recovery_refuses_detection_with_different_clothes():
    from src.kalman import ConstantVelocityKalman
    from src.recovery import colors_of, make_template, recovery_cost
    from src.track import Track
    frame = _background()
    frame[30:80, 40:60] = (40, 40, 200)
    frame[30:80, 90:110] = (40, 200, 220)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    track = Track(1, ConstantVelocityKalman(75, 55, CONFIG["kalman"]), (20.0, 50.0), (75, 55))
    track.colors = colors_of(frame, (40, 30, 20, 50), CONFIG["features"])
    track.template = make_template(gray, (40, 30, 20, 50))
    cfg = dict(CONFIG["people_tracker"], recover_max_dist_ratio=5.0)
    assert np.isfinite(recovery_cost(track, (40, 30, 20, 50), gray, frame, cfg, CONFIG["features"]))
    assert recovery_cost(track, (90, 30, 20, 50), gray, frame, cfg, CONFIG["features"]) == np.inf
