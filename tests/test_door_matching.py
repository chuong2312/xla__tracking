"""Kiểm thử ghép sự kiện qua cửa giữa hai camera theo thời gian (src/door_matching.py)."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.door import DoorCounter, DoorEvent, foot_point
from src.door_matching import match_door_events, track_links
from tools.make_synthetic_v2 import generate


def test_events_of_same_direction_close_in_time_are_linked():
    a = [DoorEvent(100, 1, "IN"), DoorEvent(200, 2, "OUT")]
    b = [DoorEvent(103, 7, "IN"), DoorEvent(198, 8, "OUT")]
    assert track_links(match_door_events(a, b, fps=30, max_gap_s=1.5)) == {1: 7, 2: 8}


def test_opposite_directions_are_never_linked():
    a = [DoorEvent(100, 1, "IN")]
    b = [DoorEvent(100, 7, "OUT")]
    assert match_door_events(a, b, fps=30, max_gap_s=1.5) == []


def test_events_too_far_apart_are_not_linked():
    a = [DoorEvent(100, 1, "IN")]
    b = [DoorEvent(200, 7, "IN")]
    assert match_door_events(a, b, fps=30, max_gap_s=1.5) == []


def test_two_people_entering_one_after_another_keep_their_order():
    # Hai người vào cách nhau 1 giây; mỗi camera ghi lệch vài frame.
    a = [DoorEvent(100, 1, "IN"), DoorEvent(130, 2, "IN")]
    b = [DoorEvent(104, 7, "IN"), DoorEvent(133, 8, "IN")]
    matches = match_door_events(a, b, fps=30, max_gap_s=1.5)
    assert track_links(matches) == {1: 7, 2: 8}
    assert matches[0].gap_s == pytest.approx(4 / 30)


def test_corridor_door_scenario_links_the_same_people(tmp_path):
    """Trên video tổng hợp hanh_lang_cua: dùng đáp án làm track, đếm cửa ở mỗi camera rồi ghép."""
    fps = 10.0
    folder = generate("hanh_lang_cua", tmp_path, fps, seed=0)
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    events = {}
    for camera, id_offset in (("cam_a", 0), ("cam_b", 100)):
        geometry = scene["cameras"][camera]
        counter = DoorCounter(geometry["door_line"], 5, geometry["door_inside_side"])
        with (folder / camera / "gt_tracking.csv").open(encoding="utf-8") as stream:
            rows = sorted(csv.DictReader(stream), key=lambda r: (int(r["frame"]), int(r["gt_id"])))
        for row in rows:
            box = [int(row[k]) for k in ("x", "y", "w", "h")]
            counter.update(int(row["frame"]), int(row["gt_id"]) + id_offset, foot_point(box))
        events[camera] = counter.events
    links = track_links(match_door_events(events["cam_a"], events["cam_b"], fps, max_gap_s=1.5))
    assert links and all(b == a + 100 for a, b in links.items())
