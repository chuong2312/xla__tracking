"""Test bước tìm người trên video thật của nhóm (EPFL phòng thí nghiệm 4 người, camera 0).

Bộ video chưa có đáp án, nên kiểm tra bằng hình học thật của camera. EPFL đo sẵn hai ma trận cho mỗi
camera: G_sàn (ảnh → bản đồ sàn) và G_đầu (ảnh → bản đồ của mặt phẳng ngang cao 1,75 m). Với khung có
chân ở điểm p:  đỉnh đầu mong đợi = G_đầu⁻¹ · G_sàn · p,  cao mong đợi = y_chân − y_đầu.
Khung đúng cỡ một người thì  cao_khung / cao_mong_đợi  gần 1.
"""
import numpy as np

from du_lieu_that import CONFIG
from src.common.homography import project
from src.common.utils import camera_video, iter_video
from src.detect.background import MedianRunningBackground
from src.detect.detection import PersonDetector
from tools.epfl_homography import read_calibration


def test_detector_boxes_have_person_height(lab4p, calibration_4p):
    """Frame 900–1300: tỉ lệ cao khung / cao mong đợi có trung vị 0,7–1,15 (đo được 0,90),
    và ít nhất 80% frame có khung (đo được 40/41)."""
    camera = read_calibration(calibration_4p)[0]
    detector = PersonDetector(CONFIG["detection"])
    ratios, frames_with_box, checked = [], 0, 0
    for index, frame in iter_video(camera_video(lab4p / "cam_a")):
        boxes = detector.detect(frame)
        if 900 <= index <= 1300 and index % 10 == 0:
            checked += 1
            frames_with_box += bool(boxes)
            for x, y, w, h in boxes:
                floor = project(camera["ground"], [(x + w / 2.0, y + h)])
                head_y = project(np.linalg.inv(camera["head"]), floor)[0][1]
                if y + h - head_y > 10:
                    ratios.append(h / (y + h - head_y))
        if index >= 1300:
            break
    assert 0.7 <= np.median(ratios) <= 1.15
    assert frames_with_box >= 0.8 * checked


def test_median_background_sees_nothing_in_the_empty_room(lab4p):
    """Nền trung vị lấy từ 30 frame đầu, lúc phòng còn trống: frame 31–60 gần như không có
    foreground (≤ 1% ảnh; đo được 0,10%)."""
    model = MedianRunningBackground(dict(CONFIG["detection"], background="median_running"))
    shares = []
    for index, frame in iter_video(camera_video(lab4p / "cam_a")):
        mask = model.apply(frame)
        if index > 30:
            shares.append((mask > 0).mean())
        if index >= 60:
            break
    assert np.mean(shares) <= 0.01
