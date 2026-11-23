"""Test bám người trên video thật của nhóm (EPFL phòng thí nghiệm 4 người; chưa có đáp án)."""
import copy
import json
import subprocess
import sys

import cv2
import numpy as np

from du_lieu_that import BLACK_WHITE_800, BROWN_800, BROWN_805, CONFIG, ROOT, read_frame
from src.common.kalman import ConstantVelocityKalman
from src.common.utils import camera_video
from src.pipeline import process_camera
from src.tracking.recovery import colors_of, make_template, recovery_cost
from src.tracking.track import Track

ACTIVE = ("confirmed", "merged", "static")


def test_tracker_keeps_few_names_on_real_video(lab4p, tmp_path):
    """1000 frame đầu camera 0 (tối đa 4 người): không quá 12 tên (đo được 7), và có ít nhất một
    người được giữ tên liền ≥ 500 frame (đo được 844)."""
    config = copy.deepcopy(CONFIG)
    config["detection"]["height_model"]["frames"] = 600
    result = process_camera(camera_video(lab4p / "cam_a"), "cam_a", tmp_path, config, max_frames=1000)
    lengths = {}
    for row in result.rows:
        if row[2] in ACTIVE:
            lengths[row[1]] = lengths.get(row[1], 0) + 1
    assert len(lengths) <= 12
    assert max(lengths.values()) >= 500


def test_recovery_refuses_detection_with_different_clothes(lab4p):
    """Frame 800: người áo nâu và người áo đen quần trắng (độ giống màu 0,48). Track mang màu người
    áo nâu không được nhận khung người kia, nhưng nhận được khung của chính mình ở frame 805 (độ giống
    0,86). Ngưỡng màu trong test đặt 0,6, giữa hai con số đó."""
    frame, later = read_frame(lab4p, "cam_a", 800), read_frame(lab4p, "cam_a", 805)
    gray, later_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(later, cv2.COLOR_BGR2GRAY)

    def track_at(box):
        x, y, w, h = box
        center = (x + w / 2, y + h / 2)
        track = Track(1, ConstantVelocityKalman(*center, CONFIG["kalman"]), (float(w), float(h)), center)
        track.colors = colors_of(frame, BROWN_800, CONFIG["features"])
        track.template = make_template(gray, BROWN_800)
        return track

    cfg = dict(CONFIG["people_tracker"], recover_min_color_similarity=0.6)
    assert recovery_cost(track_at(BLACK_WHITE_800), BLACK_WHITE_800, gray, frame, cfg, CONFIG["features"]) == np.inf
    assert np.isfinite(recovery_cost(track_at(BROWN_805), BROWN_805, later_gray, later, cfg, CONFIG["features"]))


def test_run_py_end_to_end_on_real_clip(lab4p, tmp_path):
    """run.py chạy trọn hệ thống trên 300 frame đầu của cảnh hai camera. Cảnh chưa có đáp án nên
    summary.json rỗng, nhưng vẫn ghi track và ID toàn cục."""
    out = tmp_path / "ket_qua"
    subprocess.run([sys.executable, str(ROOT / "run.py"), "--scene", str(lab4p), "--out", str(out), "--max-frames", "300"],
                   cwd=ROOT, check=True, capture_output=True)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == {}
    for name in ("cam_a/tracks_linked.csv", "cam_b/tracks_linked.csv", "global_ids_ngoai_hinh.csv", "global_ids_ngoai_hinh_san.csv"):
        assert (out / name).exists(), name
