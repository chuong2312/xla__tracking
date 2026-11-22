"""Test ổn định hình (src/detect/stabilization.py, QĐ-30) với video thật của nhóm (QĐ-31, QĐ-32).

Bộ video của nhóm quay bằng camera cố định, nên test kiểm tra rằng ổn định hình KHÔNG làm thay đổi frame.
Hiện không có video thật nào bị trôi, nên phần "đưa frame bị trôi về đúng chỗ" chỉ được kiểm tra bằng con
số (ma trận biến đổi đặt tay), không dùng ảnh.
"""
import numpy as np

from du_lieu_that import CONFIG
from src.common.utils import camera_video, iter_video
from src.detect.stabilization import Stabilizer, corner_shift


def test_fixed_camera_frames_are_left_unchanged(lab4p):
    stabilizer = Stabilizer(CONFIG["stabilization"])
    for index, frame in iter_video(camera_video(lab4p / "cam_a")):
        assert np.array_equal(stabilizer.apply(frame), frame)
        assert stabilizer.valid_mask is None
        if index == 150:
            break


def test_box_is_mapped_back_to_original_coordinates():
    """Frame bị trôi (+12, −7) px thì M dời ngược (−12, +7); box ổn định đổi về gốc phải dời (+12, −7)."""
    stabilizer = Stabilizer(CONFIG["stabilization"])
    stabilizer.matrix = np.float64([[1, 0, -12], [0, 1, 7]])
    x, y, w, h = stabilizer.to_original((100, 80, 30, 60))
    assert (x, y, w, h) == (112, 73, 30, 60)


def test_corner_shift_of_identity_is_zero():
    assert corner_shift(np.float64([[1, 0, 0], [0, 1, 0]]), 320, 240) == 0.0
    assert corner_shift(np.float64([[1, 0, 3], [0, 1, 4]]), 320, 240) == 5.0
