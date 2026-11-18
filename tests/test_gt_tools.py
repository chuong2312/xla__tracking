import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.annotate import interpolate_annotations, write_gt
from tools.verify_gt import read_gt, validate_gt

def test_interpolation_two_keys():
    assert interpolate_annotations({0:{1:(0,0,10,10)},2:{1:(2,4,12,14)}})==[(0,1,0,0,10,10),(1,1,1,2,11,12),(2,1,2,4,12,14)]
def test_interpolation_three_keys_and_single_key():
    got=interpolate_annotations({0:{1:(0,0,2,2),2:(4,4,2,2)},2:{1:(2,0,2,2)},4:{1:(4,0,2,2)}})
    assert len([r for r in got if r[1]==1])==5
    assert [r for r in got if r[1]==2]==[(0,2,4,4,2,2)]
def test_csv_round_trip(tmp_path: Path):
    path=tmp_path/"gt.csv"; rows=[(0,3,1,2,4,5)]; write_gt(path,rows); assert read_gt(path)==rows
def test_detect_out_of_frame():
    warnings=validate_gt([(0,1,-1,0,4,4),(2,1,1,1,3,3)],10,10)
    assert any("ngoài khung" in warning for warning in warnings)
    assert any("lỗ hổng" in warning for warning in warnings)
