"""Kiểm thử MOTA, MOTP, IDF1 (src/evaluation/mot_metrics.py)."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.mot_metrics import mot_metrics


def _write(path: Path, header, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _gt(path: Path, frames: int = 10) -> Path:
    rows = [(f, gid, x, 0, 10, 10) for f in range(frames) for gid, x in ((1, 0), (2, 50))]
    return _write(path, ["frame", "gt_id", "x", "y", "w", "h"], rows)


def _tracks(path: Path, rows) -> Path:
    return _write(path, ["frame", "track_id", "state", "x", "y", "w", "h"], rows)


def test_perfect_tracking_scores_one(tmp_path):
    rows = [(f, tid, "confirmed", x, 0, 10, 10) for f in range(10) for tid, x in ((7, 0), (8, 50))]
    result = mot_metrics(_tracks(tmp_path / "t.csv", rows), _gt(tmp_path / "g.csv"), 0.5)
    assert result["mota"] == pytest.approx(1.0)
    assert result["motp"] == pytest.approx(1.0)
    assert result["idf1"] == pytest.approx(1.0)
    assert result["idsw"] == 0


def test_swapping_ids_halfway_counts_two_switches(tmp_path):
    rows = []
    for f in range(10):
        first, second = (7, 8) if f < 5 else (8, 7)
        rows += [(f, first, "confirmed", 0, 0, 10, 10), (f, second, "confirmed", 50, 0, 10, 10)]
    result = mot_metrics(_tracks(tmp_path / "t.csv", rows), _gt(tmp_path / "g.csv"), 0.5)
    assert result["idsw"] == 2
    assert result["mota"] == pytest.approx(1.0 - 2 / 20)
    assert result["idf1"] == pytest.approx(0.5)


def test_missing_and_extra_boxes_count_as_fn_and_fp(tmp_path):
    rows = [(f, 7, "confirmed", 0, 0, 10, 10) for f in range(10)]
    rows += [(0, 9, "confirmed", 200, 200, 10, 10)]
    result = mot_metrics(_tracks(tmp_path / "t.csv", rows), _gt(tmp_path / "g.csv"), 0.5)
    assert result["fn"] == 10
    assert result["fp"] == 1
    assert result["mota"] == pytest.approx(1.0 - 11 / 20)


def test_tentative_tracks_are_ignored(tmp_path):
    rows = [(f, tid, "tentative", x, 0, 10, 10) for f in range(10) for tid, x in ((7, 0), (8, 50))]
    result = mot_metrics(_tracks(tmp_path / "t.csv", rows), _gt(tmp_path / "g.csv"), 0.5)
    assert result["fn"] == 20 and result["fp"] == 0
