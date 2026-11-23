"""Kiểm thử homography mặt sàn (src/common/homography.py) bằng hiệu chỉnh thật của bộ video nhóm (QĐ-31, QĐ-32).

EPFL đo sẵn cho mỗi camera ma trận G "ảnh → bản đồ sàn" (calibration-4p.txt). Lấy vài điểm trên sàn ở ảnh
camera 0, đổi ra bản đồ sàn bằng G_0 rồi vào ảnh camera 1 bằng G_1⁻¹: đó là các cặp điểm sàn thật trên hai
camera thật, đóng vai các điểm băng keo được click.
"""
import numpy as np
import pytest

from du_lieu_that import read_frame
from src.common import homography
from tools.calibrate_floor import render_check
from tools.epfl_homography import floor_to_floor, read_ground_homographies

FLOOR_PIXELS_CAM0 = [(120, 200), (200, 210), (280, 230), (160, 250), (240, 260), (100, 240), (300, 200), (180, 180)]


@pytest.fixture(scope="module")
def floor_pairs(calibration_4p):
    """Các cặp điểm sàn (camera 0, camera 1) và H_AB đúng theo hiệu chỉnh EPFL."""
    g = read_ground_homographies(calibration_4p)
    truth = floor_to_floor(g[0], g[1])
    points_a = np.array(FLOOR_PIXELS_CAM0, dtype=float)
    return points_a, homography.project(truth, points_a), truth


def test_estimate_recovers_true_floor_homography(floor_pairs):
    points_a, points_b, truth = floor_pairs
    H, errors = homography.estimate(points_a, points_b, ransac_threshold_px=3.0)
    assert errors.max() < 0.5
    probe = np.array([[180.0, 220.0]])
    assert np.allclose(homography.project(H, probe), homography.project(truth, probe), atol=1.0)


def test_one_wrong_click_is_rejected_by_ransac(floor_pairs):
    points_a, points_b, _ = floor_pairs
    points_b = points_b.copy()
    points_b[2] += (40.0, -25.0)
    _, errors = homography.estimate(points_a, points_b, ransac_threshold_px=3.0)
    assert errors[2] > 20.0
    assert np.delete(errors, 2).max() < 1.0


def test_needs_four_matching_points(floor_pairs):
    points_a, points_b, _ = floor_pairs
    with pytest.raises(ValueError, match="ít nhất 4"):
        homography.estimate(points_a[:3], points_b[:3], 3.0)
    with pytest.raises(ValueError, match="bằng nhau"):
        homography.estimate(points_a, points_b[:5], 3.0)


def test_save_load_round_trip_and_check_image(floor_pairs, lab4p, tmp_path):
    points_a, points_b, _ = floor_pairs
    H, errors = homography.estimate(points_a, points_b, 3.0)
    path = tmp_path / "homography.json"
    homography.save(path, H, points_a, points_b, (360, 288), errors)
    loaded = homography.load(path)
    assert np.allclose(loaded["H_ab"], H)
    frame = read_frame(lab4p, "cam_b", 0)
    image = render_check(frame, loaded["H_ab"], loaded["points_a"], loaded["points_b"], errors)
    assert image.shape == frame.shape and not np.array_equal(image, frame)
