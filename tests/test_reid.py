"""Kiểm thử đặc trưng ngoại hình và nhận dạng lại (src/reid/) trên video thật của nhóm (QĐ-31, QĐ-32).

Ảnh lấy từ camera 0 của bộ EPFL phòng thí nghiệm 4 người; khung người được chọn bằng mắt (tests/du_lieu_that.py).
Các test gallery, geometry, assign_global_ids chỉ có con số nên không cần video. Chế độ "oracle" (dùng
khung đáp án làm track) cần đáp án, mà bộ video hiện chưa có, nên không có test oracle.
"""
import numpy as np
import pytest

from du_lieu_that import (BLACK_WHITE_800, BROWN_800, BROWN_805, CONFIG, FLAT_REGION_0, MAN_BLACK_WHITE_700,
                          MAN_BLACK_WHITE_710, WOMAN_BLACK_700, crop, read_frame)
from src.reid.color_features import apply_gains, body_histograms, color_similarity, gray_world_gains
from src.reid.gallery import Observation, select_gallery
from src.reid.keypoint_features import extract, normalize_crop
from src.reid.reid import assign_global_ids
from src.reid.reid_similarity import combine, geometry_similarity, keypoint_similarity

FEATURES = CONFIG["features"]
REID = CONFIG["reid"]


def test_gray_world_equalises_channel_means_on_real_frame(lab4p):
    frame = read_frame(lab4p, "cam_a", 700)
    balanced = apply_gains(frame, gray_world_gains(frame)).reshape(-1, 3).mean(axis=0)
    assert balanced.max() - balanced.min() < 2.0


def test_same_person_is_more_similar_than_different_people(lab4p):
    """Cùng người cách vài frame giống màu 0,82 và 0,86; hai người khác nhau 0,64 (cùng áo đen) và 0,48."""
    def colors(index, box):
        return body_histograms(crop(read_frame(lab4p, "cam_a", index), box), None, FEATURES)

    same = [color_similarity(colors(700, MAN_BLACK_WHITE_700), colors(710, MAN_BLACK_WHITE_710)),
            color_similarity(colors(800, BROWN_800), colors(805, BROWN_805))]
    different = [color_similarity(colors(700, MAN_BLACK_WHITE_700), colors(700, WOMAN_BLACK_700)),
                 color_similarity(colors(800, BLACK_WHITE_800), colors(800, BROWN_800))]
    assert min(same) > max(different)


def test_normalize_crop_keeps_ratio_and_masks_padding(lab4p):
    piece, mask = normalize_crop(read_frame(lab4p, "cam_a", 700), MAN_BLACK_WHITE_700, FEATURES["crop_size"])
    assert piece.shape == (256, 128, 3)
    assert mask[:, 0].max() == 0 and mask[128, 64] == 255


@pytest.mark.parametrize("method", ["sift", "orb", "sift_auto", "asift"])
def test_every_method_respects_max_keypoints_and_mask(lab4p, method):
    piece, mask = normalize_crop(read_frame(lab4p, "cam_a", 700), MAN_BLACK_WHITE_700, FEATURES["crop_size"])
    features = extract(piece, mask, method, FEATURES)
    assert 0 < len(features.points) <= FEATURES["max_keypoints"]
    assert features.points[:, 0].min() >= np.flatnonzero(mask[128])[0] - 1


def test_sift_auto_retries_on_low_texture_real_region(lab4p):
    """Vùng sàn trơn ở frame 0: SIFT thường được 5 điểm, sift_auto hạ ngưỡng tương phản 2 lần và được 188 điểm."""
    piece, mask = normalize_crop(read_frame(lab4p, "cam_a", 0), FLAT_REGION_0, FEATURES["crop_size"])
    plain, auto = extract(piece, mask, "sift", FEATURES), extract(piece, mask, "sift_auto", FEATURES)
    assert auto.retries > 0
    assert len(auto.points) > len(plain.points)


def test_keypoint_similarity_needs_enough_matches(lab4p):
    frame = read_frame(lab4p, "cam_a", 700)
    same = extract(*normalize_crop(frame, MAN_BLACK_WHITE_700, FEATURES["crop_size"]), "sift", FEATURES)
    other = extract(*normalize_crop(frame, WOMAN_BLACK_700, FEATURES["crop_size"]), "sift", FEATURES)
    assert keypoint_similarity([same], [same], "sift", REID) == pytest.approx(1.0)
    assert keypoint_similarity([same], [other], "sift", REID) == 0.0


def test_gallery_prefers_large_clean_spaced_boxes():
    obs = [Observation(f, (100, 100, 10 + f, 40), clean=f != 50) for f in range(0, 60)]
    obs.append(Observation(70, (0, 100, 99, 99), clean=True))
    chosen = select_gallery(obs, (640, 360), size=3, min_gap_frames=15, border_px=2)
    assert [o.frame for o in chosen] == [29, 44, 59]


def test_geometry_and_combine():
    identity = np.eye(3)
    assert geometry_similarity({1: (10, 10)}, {1: (10, 10)}, identity, 25.0) == pytest.approx(1.0)
    assert geometry_similarity({1: (10, 10)}, {2: (10, 10)}, identity, 25.0) is None
    weights = {"keypoint": 0.4, "mau": 0.4, "san": 0.2}
    assert combine({"keypoint": 1.0, "mau": 0.0, "san": None}, weights) == pytest.approx(0.5)


def test_assign_global_ids_marks_unmatched_tracks():
    rows = assign_global_ids([1, 2], [5, 6], np.array([[0.9, 0.1], [0.1, 0.2]]), threshold=0.3)
    assert ("cam_a", 1, 1) in rows and ("cam_b", 5, 1) in rows
    assert ("cam_a", 2, -1) in rows and ("cam_b", 6, -1) in rows
