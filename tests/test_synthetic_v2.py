"""Kiểm thử video tổng hợp v2 (tools/make_synthetic_v2.py), chạy ở 5 fps cho nhanh."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.homography import project
from tools.make_synthetic_v2 import generate
from tools.synthetic_render import person_patch
from tools.synthetic_scenarios import SCENARIOS
from tools.synthetic_scene import IMAGE_SIZE

FPS = 5.0


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def scene_dirs(tmp_path_factory):
    first = generate("s2", tmp_path_factory.mktemp("a"), FPS, seed=3)
    second = generate("s2", tmp_path_factory.mktemp("b"), FPS, seed=3)
    return first, second


def test_same_seed_gives_same_ground_truth(scene_dirs):
    first, second = scene_dirs
    for name in ("cam_a/gt_tracking.csv", "cam_b/gt_tracking.csv", "gt_reid.csv"):
        assert (first / name).read_text() == (second / name).read_text()


def test_boxes_stay_inside_image_and_use_new_columns(scene_dirs):
    width, height = IMAGE_SIZE
    for camera in ("cam_a", "cam_b"):
        rows = _rows(scene_dirs[0] / camera / "gt_tracking.csv")
        assert rows and set(rows[0]) == {"frame", "gt_id", "x", "y", "w", "h", "occluded", "keyframe"}
        for row in rows:
            x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
            assert x >= 0 and y >= 0 and x + w <= width and y + h <= height
            assert row["keyframe"] == "1" and row["occluded"] in {"0", "1", "2"}


def test_scene_homography_maps_tape_marks_from_a_to_b(scene_dirs):
    scene = json.loads((scene_dirs[0] / "scene.json").read_text())
    marks_a = scene["cameras"]["cam_a"]["tape_marks"]
    marks_b = np.array(scene["cameras"]["cam_b"]["tape_marks"])
    assert np.allclose(project(np.array(scene["H_ab"]), marks_a), marks_b, atol=0.1)


def test_crossing_scenario_contains_occlusion(scene_dirs):
    levels = [int(row["occluded"]) for row in _rows(scene_dirs[0] / "cam_a" / "gt_tracking.csv")]
    assert max(levels) >= 1


def test_static_person_stands_still_for_five_seconds():
    person = SCENARIOS["s3"].people[0]
    positions = {person.position(t) for t in np.arange(4.5, 9.51, 0.5)}
    assert len(positions) == 1


def test_same_colour_pair_looks_identical():
    first, second = SCENARIOS["s4"].people[0], SCENARIOS["s4"].people[1]
    assert np.array_equal(person_patch(first, 20, 60), person_patch(second, 20, 60))
