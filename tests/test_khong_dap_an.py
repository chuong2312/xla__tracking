"""Kiểm thử bài chấm không cần đáp án (src/evaluation/khong_dap_an.py, QĐ-34). Chỉ có con số (khung, frame)."""
from src.evaluation.khong_dap_an import coverage, ghost_share, jumps


def _track(frame, track_id, box, state="confirmed"):
    x, y, w, h = box
    return {"frame": str(frame), "track_id": str(track_id), "state": state, "x": str(x), "y": str(y), "w": str(w), "h": str(h)}


def _detection(frame, box):
    x, y, w, h = box
    return {"frame": str(frame), "x": str(x), "y": str(y), "w": str(w), "h": str(h)}


def test_coverage_counts_detections_with_a_track_on_them():
    detections = [_detection(1, (10, 10, 20, 60)), _detection(1, (100, 10, 20, 60))]
    tracks = [_track(1, 7, (11, 10, 20, 60))]
    assert coverage(detections, tracks, None, 288) == 0.5


def test_ghost_share_counts_confirmed_tracks_without_detection():
    detections = [_detection(1, (10, 10, 20, 60))]
    tracks = [_track(1, 7, (10, 10, 20, 60)), _track(1, 8, (200, 10, 20, 60))]
    assert ghost_share(detections, tracks) == 0.5


def test_jumps_flags_teleporting_track_but_not_walking():
    walking = [_track(f, 1, (10 + 2 * f, 10, 20, 60)) for f in range(5)]
    teleport = [_track(0, 2, (10, 10, 20, 60)), _track(1, 2, (200, 10, 20, 60))]
    assert jumps(walking) == 0
    assert jumps(teleport) == 1
