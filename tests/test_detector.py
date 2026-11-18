"""Test detector bằng frame tổng hợp tạo từ NumPy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from src.detector import Detector

def make_config(**updates):
    detector = {"blur_kernel": 1, "history": 30, "varThreshold": 16.0, "learningRate": 0.001,
        "opening_kernel": 1, "opening_iterations": 0, "closing_kernel": 1, "closing_iterations": 0,
        "dilation_kernel": 1, "dilation_iterations": 0, "min_area": 40, "max_area": 10000,
        "min_ratio": 0.2, "max_ratio": 5.0, "use_hog_baseline": False}
    detector.update(updates.pop("detector", {}))
    scene = {"resize_width": 160, "resize_height": 120, "roi": None, "ignore_regions": []}
    scene.update(updates.pop("scene", {}))
    return {"detector": detector, "scene": scene}

def sequence(ignore_regions=None):
    rng = np.random.default_rng(13)
    background = rng.integers(30, 90, (120, 160, 3), dtype=np.uint8)
    frames = [background.copy() for _ in range(12)]
    moving = background.copy()
    cv2.rectangle(moving, (20, 20), (39, 69), (240, 240, 240), -1)
    cv2.rectangle(moving, (85, 25), (109, 64), (180, 180, 180), -1)
    frames.extend([moving.copy() for _ in range(4)])
    return frames

def warm_detect(detector, frames):
    boxes = []
    for frame in frames: boxes = detector.detect(frame)
    return boxes

def test_detects_moving_regions_and_returns_binary_mask():
    detector = Detector(make_config())
    boxes = warm_detect(detector, sequence())
    assert len(boxes) == 2
    mask = detector.get_foreground_mask()
    assert mask.dtype == np.uint8 and mask.shape == (120,160)
    assert set(np.unique(mask)).issubset({0,255})
    assert all(0 <= x and 0 <= y and x+w <= 160 and y+h <= 120 for x,y,w,h in boxes)

def test_shadow_value_127_is_not_kept(monkeypatch):
    detector = Detector(make_config())
    class ShadowModel:
        def apply(self, *args, **kwargs): return np.full((120,160),127,np.uint8)
    detector._mog2 = ShadowModel()
    assert detector.detect(np.zeros((120,160,3),np.uint8)) == []
    assert not np.any(detector.get_foreground_mask())

def test_area_and_aspect_filters():
    frames=sequence()
    small=Detector(make_config(detector={"min_area":5000}))
    assert warm_detect(small,frames)==[]
    ratio=Detector(make_config(detector={"min_ratio":0.5,"max_ratio":0.8}))
    assert warm_detect(ratio,frames)==[]

def test_roi_and_ignore_regions_remove_detections_and_mask_pixels():
    frames=sequence()
    roi_detector=Detector(make_config(scene={"roi":[0,0,70,120]}))
    roi_boxes=warm_detect(roi_detector,frames)
    assert len(roi_boxes)==1
    ignored=Detector(make_config(scene={"ignore_regions":[[0,0,60,100]]}))
    ignored_boxes=warm_detect(ignored,frames)
    assert len(ignored_boxes)==1
    assert np.all(ignored.get_foreground_mask()[:, :60] == 0)

def test_ignore_regions_outside_frame_are_safe():
    detector=Detector(make_config(scene={"ignore_regions":[[-20,-20,30,30],[500,500,20,20]]}))
    assert warm_detect(detector,sequence())

def test_seeded_sequence_is_deterministic():
    frames=sequence()
    first=warm_detect(Detector(make_config()),frames)
    second=warm_detect(Detector(make_config()),frames)
    assert first==second
