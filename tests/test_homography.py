"""Kiểm thử homography mặt sàn (src/homography.py) bằng hình học biết trước của video tổng hợp v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import homography
from tools.calibrate_floor import render_check
from tools.synthetic_scenarios import LAYOUTS
from tools.synthetic_scene import floor_to_image

LAYOUT = LAYOUTS["san_chung"]


def _tape_points():
    marks = LAYOUT.tape_marks_m
    return homography.project(floor_to_image(LAYOUT, "cam_a"), marks), homography.project(floor_to_image(LAYOUT, "cam_b"), marks)


def test_estimate_recovers_true_floor_homography():
    points_a, points_b = _tape_points()
    H, errors = homography.estimate(points_a, points_b, ransac_threshold_px=3.0)
    assert errors.max() < 1e-3
    floor_point = [(2.5, 1.7)]
    truth = homography.project(floor_to_image(LAYOUT, "cam_b"), floor_point)
    via_a = homography.project(H, homography.project(floor_to_image(LAYOUT, "cam_a"), floor_point))
    assert np.allclose(via_a, truth, atol=0.5)


def test_one_wrong_click_is_rejected_by_ransac():
    points_a, points_b = _tape_points()
    points_b[2] += (40.0, -25.0)
    _, errors = homography.estimate(points_a, points_b, ransac_threshold_px=3.0)
    assert errors[2] > 20.0
    assert np.delete(errors, 2).max() < 1.0


def test_needs_four_matching_points():
    points_a, points_b = _tape_points()
    with pytest.raises(ValueError, match="ít nhất 4"):
        homography.estimate(points_a[:3], points_b[:3], 3.0)
    with pytest.raises(ValueError, match="bằng nhau"):
        homography.estimate(points_a, points_b[:5], 3.0)


def test_save_load_round_trip_and_check_image(tmp_path):
    points_a, points_b = _tape_points()
    H, errors = homography.estimate(points_a, points_b, 3.0)
    path = tmp_path / "homography.json"
    homography.save(path, H, points_a, points_b, (640, 360), errors)
    loaded = homography.load(path)
    assert np.allclose(loaded["H_ab"], H)
    image = render_check(np.zeros((360, 640, 3), np.uint8), loaded["H_ab"], loaded["points_a"], loaded["points_b"], errors)
    assert image.shape == (360, 640, 3) and image.any()
