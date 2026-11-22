import csv
from pathlib import Path
import cv2
from tools.make_synthetic import generate
from src.common.utils import iter_video

def test_synthetic_outputs(tmp_path: Path):
    generate("cross",5,(160,120),0,tmp_path)
    frames=list(iter_video(tmp_path/"cam_a"/"video.mp4"))
    assert len(frames)==5
    with (tmp_path/"gt_tracking.csv").open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    assert len(rows)==15 and {r["gt_id"] for r in rows}=={"0","1","2"}
    assert cv2.imread(str(tmp_path/"cam_b"/"video.mp4")) is None
