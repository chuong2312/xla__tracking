"""Kiểm thử đếm qua cửa (src/door.py), gồm test hồi quy bắt buộc IN 1 / OUT 1."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.door import DoorCounter, foot_point, signed_distance
from tools.make_synthetic_v2 import camera_geometry
from tools.synthetic_scenarios import SCENARIOS
from tools.synthetic_scene import door_events, floor_to_image, n_frames, person_box

LINE = (0.0, 0.0, 100.0, 0.0)


def test_signed_distance_sign_follows_line_direction():
    # Trục y của ảnh hướng xuống: điểm có y > 0 nằm bên phải khi đi từ (0,0) sang (100,0).
    assert signed_distance(LINE, (50, 10)) == pytest.approx(10.0)
    assert signed_distance(LINE, (50, -4)) == pytest.approx(-4.0)


def test_regression_one_in_one_out_uses_only_positions():
    """Hồi quy bắt buộc: một người vào, một người ra thì phải ra đúng IN 1 / OUT 1.

    Chỉ đưa vào điểm chân, không có thông tin màu nào.
    """
    counter = DoorCounter(LINE, margin_px=5, inside_side="positive")
    for frame, y in enumerate(range(-20, 21, 2)):
        counter.update(frame, 1, (50, y))
    for frame, y in enumerate(range(20, -21, -2)):
        counter.update(frame, 2, (30, y))
    assert counter.counts() == {"IN": 1, "OUT": 1}
    assert [(e.track_id, e.direction) for e in counter.events] == [(1, "IN"), (2, "OUT")]


def test_jitter_inside_margin_is_not_counted():
    counter = DoorCounter(LINE, margin_px=5, inside_side="positive")
    for frame, y in enumerate([-10, 3, -3, 4, -4, 2, -2]):
        counter.update(frame, 1, (50, y))
    assert counter.events == []
    counter.update(10, 1, (50, 10))
    for frame, y in enumerate([4, -4, 3, -3], start=11):
        counter.update(frame, 1, (50, y))
    assert counter.counts() == {"IN": 1, "OUT": 0}


def test_crossing_outside_door_width_is_ignored():
    counter = DoorCounter(LINE, margin_px=5, inside_side="positive")
    counter.update(0, 1, (150, -10))
    assert counter.update(1, 1, (150, 10)) is None


def test_finish_track_forgets_side():
    counter = DoorCounter(LINE, margin_px=5, inside_side="positive")
    counter.update(0, 7, (50, -10))
    counter.finish_track(7)
    assert counter.update(1, 7, (50, 10)) is None


@pytest.mark.parametrize("scenario_name", ["door", "hanh_lang_cua"])
@pytest.mark.parametrize("camera", ["cam_a", "cam_b"])
def test_synthetic_door_scenarios_match_ground_truth(scenario_name, camera):
    """door: 2 người qua cửa cùng lúc, ngược chiều. hanh_lang_cua: cửa ở tường bên, có người đi ngang không vào."""
    fps = 30.0
    scenario = SCENARIOS[scenario_name]
    geometry = camera_geometry(scenario.layout, camera)
    counter = DoorCounter(geometry["door_line"], margin_px=5, inside_side=geometry["door_inside_side"])
    H = floor_to_image(scenario.layout, camera)
    for frame in range(n_frames(fps)):
        for person in scenario.people:
            position = person.position(frame / fps)
            if position is not None:
                counter.update(frame, person.gt_id, foot_point(person_box(H, *position)))
    expected = door_events(scenario, fps)
    assert Counter((e.track_id, e.direction) for e in counter.events) == Counter((g, d) for _, g, d in expected)
    for event, (frame, _, _) in zip(sorted(counter.events, key=lambda e: e.track_id), sorted(expected, key=lambda e: e[1])):
        assert abs(event.frame - frame) <= 10
