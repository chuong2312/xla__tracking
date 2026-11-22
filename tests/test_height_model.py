"""Test mô hình chiều cao người theo vị trí chân và việc gộp mảnh dựa trên nó (QĐ-26)."""
import numpy as np

from src.detect.detection import group_vertical_pieces
from src.detect.height_model import HeightModel, fit_height_model, theil_sen

CFG = {"bin_px": 20, "quantile": 90, "min_count": 5}


def test_fit_recovers_line_from_whole_people_and_fragments():
    rng = np.random.default_rng(0)
    boxes = []
    for foot in rng.uniform(100, 300, 400):
        height = 0.5 * foot + 10
        if rng.random() < 0.5:
            height *= rng.uniform(0.3, 0.6)  # mảnh người: thấp hơn người nguyên vẹn
        boxes.append((0, int(foot - height), 10, int(height)))
    model = fit_height_model(boxes, CFG)
    assert model is not None
    assert abs(model.expected(200) - 110) < 12


def test_fit_returns_none_without_enough_data():
    assert fit_height_model([(0, 0, 10, 50)] * 3, CFG) is None


def test_groups_head_and_legs_of_one_person():
    heights = HeightModel(a=0.5, b=0.0, points=[])  # chân ở y=200 → người cao 100
    head, legs = (50, 100, 20, 25), (48, 150, 24, 50)
    assert group_vertical_pieces([head, legs], 1.0, heights, 1.15) == [(48, 100, 24, 100)]


def test_does_not_group_two_people_one_behind_the_other():
    heights = HeightModel(a=0.5, b=0.0, points=[])
    far_person, near_person = (55, 60, 15, 40), (50, 102, 25, 98)  # gộp lại cao 140 > 1.15 × 100
    assert sorted(group_vertical_pieces([far_person, near_person], 1.0, heights, 1.15)) == sorted([far_person, near_person])


def test_theil_sen_ignores_one_outlier():
    points = [(y, 2.0 * y + 5) for y in range(10, 110, 10)] + [(120, 20.0)]
    a, b = theil_sen(points)
    assert abs(a - 2.0) < 1e-6 and abs(b - 5.0) < 1e-6


def test_tall_enough_rejects_head_fragment_but_accepts_whole_person():
    from src.detect.height_model import tall_enough
    heights = HeightModel(a=0.5, b=0.0, points=[])  # chân ở y=200 → người cao 100
    shape = (288, 360)
    assert tall_enough((100, 100, 25, 100), heights, 0.6, shape, 5)
    assert not tall_enough((100, 160, 25, 40), heights, 0.6, shape, 5)


def test_tall_enough_accepts_short_box_at_image_border():
    from src.detect.height_model import tall_enough
    heights = HeightModel(a=0.5, b=0.0, points=[])
    assert tall_enough((100, 250, 25, 38), heights, 0.6, (288, 360), 5)  # đáy box chạm mép dưới


def test_tracker_does_not_start_track_from_fragment():
    import numpy as np
    import yaml
    from src.tracking.people_tracker import PeopleTracker
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    heights = HeightModel(a=0.5, b=0.0, points=[])
    tracker = PeopleTracker(cfg["people_tracker"], cfg["kalman"], cfg["features"], heights=heights)
    frame = np.random.default_rng(0).integers(0, 255, (288, 360, 3), dtype=np.uint8)
    whole, fragment = (50, 100, 25, 100), (200, 160, 25, 40)
    tracks = tracker.update(frame, [whole, fragment], None)
    assert [t.box for t in tracks] == [whole]
