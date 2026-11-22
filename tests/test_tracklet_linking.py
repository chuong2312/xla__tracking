"""Kiểm thử nối các đoạn track bị đứt (src/tracking/tracklet_linking.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reid.color_features import body_histograms
from src.tracking.tracklet_linking import link_tracklets, relabel, segments_from_rows

CFG = {"max_gap_frames": 120, "velocity_frames": 5, "max_dist_ratio": 3.0, "min_color_similarity": 0.5,
       "weight_motion": 0.5, "weight_color": 0.5, "max_cost": 1.0}
FEATURES = {"hist_bins": [16, 16], "upper_body": [0.15, 0.55], "lower_body": [0.55, 1.0]}


def _colors(top, bottom):
    image = np.zeros((100, 40, 3), np.uint8)
    image[:50] = top
    image[50:] = bottom
    return body_histograms(image, None, FEATURES)


RED = _colors((40, 40, 200), (150, 60, 20))
GREEN = _colors((60, 170, 60), (90, 90, 90))


def _walk(track_id, frames, x0, speed):
    return [(f, track_id, "confirmed", int(x0 + speed * (f - frames[0])), 50, 20, 60, 10) for f in frames]


def test_two_pieces_of_one_walk_are_linked():
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 100, 3)
    mapping = link_tracklets(segments_from_rows(rows), {1: RED, 2: RED}, CFG)
    assert mapping == {1: 1, 2: 1}
    assert {row[1] for row in relabel(rows, mapping)} == {1}


def test_overlapping_in_time_are_never_linked():
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(10, 30), 40, 3)
    assert link_tracklets(segments_from_rows(rows), {1: RED, 2: RED}, CFG) == {1: 1, 2: 2}


def test_different_clothes_are_not_linked():
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 100, 3)
    assert link_tracklets(segments_from_rows(rows), {1: RED, 2: GREEN}, CFG) == {1: 1, 2: 2}


def test_far_away_start_is_not_linked():
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 600, 3)
    assert link_tracklets(segments_from_rows(rows), {1: RED, 2: RED}, CFG) == {1: 1, 2: 2}


def test_chain_of_three_pieces_gets_one_id():
    rows = _walk(1, range(0, 20), 10, 3) + _walk(2, range(30, 50), 100, 3) + _walk(3, range(60, 80), 190, 3)
    mapping = link_tracklets(segments_from_rows(rows), {1: RED, 2: RED, 3: RED}, CFG)
    assert mapping == {1: 1, 2: 1, 3: 1}
