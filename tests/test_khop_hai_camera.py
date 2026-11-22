"""Kiểm thử kiểm tra ghép hai camera bằng vị trí sàn (src/evaluation/khop_hai_camera.py, QĐ-35).

Chỉ có con số: khung, frame và một ma trận sàn đơn vị (bản đồ sàn trùng tọa độ ảnh).
"""
import numpy as np

from src.evaluation.khop_hai_camera import TAU, check_pairs, floor_tracks, geometric_pairs, judge, pair_distance, read_global_pairs

IDENTITY = np.eye(3)


def _row(frame, track_id, box, state="confirmed"):
    x, y, w, h = box
    return {"frame": str(frame), "track_id": str(track_id), "state": state, "x": str(x), "y": str(y), "w": str(w), "h": str(h)}


def test_floor_tracks_uses_foot_point_and_drops_cut_or_merged_boxes():
    rows = [_row(1, 7, (10, 20, 20, 60)), _row(2, 7, (10, 230, 20, 58)), _row(3, 7, (10, 20, 20, 60), "merged")]
    tracks = floor_tracks(rows, IDENTITY, 288)
    assert list(tracks[7]) == [1]
    assert np.allclose(tracks[7][1], (20.0, 80.0))


def test_pair_distance_is_median_over_common_frames():
    a = {f: np.array([0.0, 0.0]) for f in range(5)}
    b = {f: np.array([3.0, 4.0]) for f in range(2, 9)}
    assert pair_distance(a, b) == (5.0, 3)
    assert pair_distance(a, {}) == (None, 0)


def test_judge_needs_enough_common_frames():
    assert judge(5.0, 30) == "khop"
    assert judge(TAU + 1, 30) == "lech"
    assert judge(5.0, 3) == "khong_kiem_duoc"


def test_check_pairs_counts_matches_against_floor_positions():
    near = {f: np.array([100.0, 100.0]) for f in range(40)}
    far = {f: np.array([100.0 + 3 * TAU, 100.0]) for f in range(40)}
    later = {f: np.array([100.0, 100.0]) for f in range(100, 140)}
    tracks_a = {1: near, 2: near}
    tracks_b = {10: near, 20: far, 30: later}
    sure = geometric_pairs(tracks_a, tracks_b)
    assert sure == {(1, 10), (2, 10)}
    counts = check_pairs([(1, 10), (2, 20), (3, 30)], tracks_a, tracks_b, sure)
    assert counts == {"ghep": 3, "khop": 1, "lech": 1, "khong_kiem_duoc": 1, "cap_hinh_hoc": 2, "bat_duoc": 1}


def test_read_global_pairs_skips_unmatched_tracks(tmp_path):
    path = tmp_path / "global_ids_ngoai_hinh.csv"
    path.write_text("cam,track_id,global_id\ncam_a,1,1\ncam_a,2,-1\ncam_b,5,1\ncam_b,6,-1\n", encoding="utf-8")
    assert read_global_pairs(path) == [(1, 5)]
