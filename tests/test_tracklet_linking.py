"""Kiểm thử nối các đoạn track bị đứt (src/tracking/tracklet_linking.py).

Màu áo lấy từ hai người thật trong video của nhóm (camera 0, frame 800): người áo nâu và người áo đen quần
trắng, độ giống màu 0,48 (cùng người áo nâu cách 5 frame giống 0,86). Ngưỡng màu trong CFG của test đặt
0,6, giữa hai con số đó (QĐ-31, QĐ-32: chỉ dùng video thật của nhóm).
"""
import pytest

from du_lieu_that import BLACK_WHITE_800, BROWN_800, FEATURES_CFG, crop, read_frame
from src.reid.color_features import body_histograms
from src.tracking.tracklet_linking import link_tracklets, relabel, segments_from_rows

CFG = {"max_gap_frames": 120, "velocity_frames": 5, "max_dist_ratio": 3.0, "min_color_similarity": 0.6,
       "weight_motion": 0.5, "weight_color": 0.5, "max_cost": 1.0}


@pytest.fixture(scope="module")
def clothes(lab4p):
    """(màu người áo nâu, màu người áo đen quần trắng) ở frame 800 camera 0."""
    frame = read_frame(lab4p, "cam_a", 800)
    return tuple(body_histograms(crop(frame, box), None, FEATURES_CFG) for box in (BROWN_800, BLACK_WHITE_800))


def _walk(track_id, frames, x0, speed):
    return [(f, track_id, "confirmed", int(x0 + speed * (f - frames[0])), 50, 20, 60, 10) for f in frames]


def test_two_pieces_of_one_walk_are_linked(clothes):
    red, green = clothes
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 100, 3)
    mapping = link_tracklets(segments_from_rows(rows), {1: red, 2: red}, CFG)
    assert mapping == {1: 1, 2: 1}
    assert {row[1] for row in relabel(rows, mapping)} == {1}


def test_overlapping_in_time_are_never_linked(clothes):
    red, green = clothes
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(10, 30), 40, 3)
    assert link_tracklets(segments_from_rows(rows), {1: red, 2: red}, CFG) == {1: 1, 2: 2}


def test_different_clothes_are_not_linked(clothes):
    red, green = clothes
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 100, 3)
    assert link_tracklets(segments_from_rows(rows), {1: red, 2: green}, CFG) == {1: 1, 2: 2}


def test_far_away_start_is_not_linked(clothes):
    red, green = clothes
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 600, 3)
    assert link_tracklets(segments_from_rows(rows), {1: red, 2: red}, CFG) == {1: 1, 2: 2}


def test_chain_of_three_pieces_gets_one_id(clothes):
    red, green = clothes
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 100, 3) + _walk(3, range(60, 80), 190, 3)
    mapping = link_tracklets(segments_from_rows(rows), {1: red, 2: red, 3: red}, CFG)
    assert mapping == {1: 1, 2: 1, 3: 1}
