import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.annotate import cycle_occlusion, interpolate_annotations, write_gt
from tools.verify_gt import read_gt, validate_gt


def test_interpolation_two_keys():
    got = interpolate_annotations({0: {1: (0, 0, 10, 10)}, 2: {1: (2, 4, 12, 14)}})
    assert got == [(0, 1, 0, 0, 10, 10, 0, 1), (1, 1, 1, 2, 11, 12, 0, 0), (2, 1, 2, 4, 12, 14, 0, 1)]


def test_interpolation_three_keys_and_single_key():
    got = interpolate_annotations({0: {1: (0, 0, 2, 2), 2: (4, 4, 2, 2)}, 2: {1: (2, 0, 2, 2)}, 4: {1: (4, 0, 2, 2)}})
    assert len([r for r in got if r[1] == 1]) == 5
    assert [r for r in got if r[1] == 2] == [(0, 2, 4, 4, 2, 2, 0, 1)]


def test_interpolated_frames_take_larger_occlusion_level():
    boxes = {0: {1: (0, 0, 10, 10)}, 4: {1: (4, 0, 10, 10)}}
    got = interpolate_annotations(boxes, {0: {1: 0}, 4: {1: 2}})
    assert [(r[0], r[6], r[7]) for r in got] == [(0, 0, 1), (1, 2, 0), (2, 2, 0), (3, 2, 0), (4, 2, 1)]


def test_cycle_occlusion_wraps_around():
    assert [cycle_occlusion(level) for level in (0, 1, 2)] == [1, 2, 0]


def test_csv_round_trip(tmp_path: Path):
    path = tmp_path / "gt.csv"
    rows = [(0, 3, 1, 2, 4, 5, 1, 1), (1, 3, 2, 2, 4, 5, 1, 0)]
    write_gt(path, rows)
    assert read_gt(path) == rows


def test_read_old_format_uses_defaults(tmp_path: Path):
    path = tmp_path / "old.csv"
    path.write_text("frame,gt_id,x,y,w,h\n0,3,1,2,4,5\n", encoding="utf-8")
    assert read_gt(path) == [(0, 3, 1, 2, 4, 5, 0, 1)]


def test_detect_out_of_frame_holes_and_bad_values():
    rows = [(0, 1, -1, 0, 4, 4, 0, 1), (2, 1, 1, 1, 3, 3, 5, 1), (3, 1, 1, 1, 3, 3, 0, 7)]
    warnings = validate_gt(rows, 10, 10)
    assert any("ngoài khung" in warning for warning in warnings)
    assert any("lỗ hổng" in warning for warning in warnings)
    assert any("occluded phải là" in warning for warning in warnings)
    assert any("keyframe phải là" in warning for warning in warnings)
