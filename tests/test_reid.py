"""Kiểm thử đặc trưng ngoại hình và nhận dạng lại (src/color_features.py, keypoint_features.py,
gallery.py, reid_similarity.py, reid.py) và chạy oracle trên video tổng hợp v2."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.color_features import apply_gains, body_histograms, color_similarity, gray_world_gains, mean_histograms
from src.gallery import Observation, select_gallery
from src.keypoint_features import extract, normalize_crop
from src.reid import assign_global_ids
from src.reid_similarity import combine, geometry_similarity, keypoint_similarity
from tools.make_synthetic_v2 import generate
from tools.run_reid import run

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
FEATURES = CONFIG["features"]
REID = CONFIG["reid"]


def _person_image(top, bottom, seed=0, size=(60, 150)):
    rng = np.random.default_rng(seed)
    w, h = size
    image = np.zeros((h, w, 3), np.uint8)
    image[: h // 2] = top
    image[h // 2:] = bottom
    noise = rng.integers(-40, 40, (h, w, 1))
    return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)


def test_gray_world_removes_a_colour_cast():
    image = np.full((20, 20, 3), (100, 120, 160), np.uint8)
    balanced = apply_gains(image, gray_world_gains(image))
    assert np.ptp(balanced.reshape(-1, 3).mean(axis=0)) < 2


def test_same_clothes_are_similar_and_different_clothes_are_not():
    red_blue = body_histograms(_person_image((40, 40, 200), (150, 60, 20)), None, FEATURES)
    red_blue_again = body_histograms(_person_image((40, 40, 200), (150, 60, 20), seed=1), None, FEATURES)
    green_grey = body_histograms(_person_image((60, 170, 60), (90, 90, 90)), None, FEATURES)
    assert color_similarity(red_blue, red_blue_again) > 0.6
    assert color_similarity(red_blue, green_grey) < 0.2
    merged = mean_histograms([red_blue, red_blue_again])
    assert merged["upper"].sum() == pytest.approx(1.0, abs=1e-4)


def test_normalize_crop_keeps_ratio_and_masks_padding():
    frame = np.full((300, 300, 3), 200, np.uint8)
    crop, mask = normalize_crop(frame, (10, 10, 50, 200), FEATURES["crop_size"])
    assert crop.shape == (256, 128, 3)
    assert mask[:, 0].max() == 0 and mask[128, 64] == 255


@pytest.mark.parametrize("method", ["sift", "orb", "sift_auto", "asift"])
def test_every_method_respects_max_keypoints_and_mask(method):
    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8), (3, 3), 0)
    crop, mask = normalize_crop(frame, (20, 20, 60, 220), FEATURES["crop_size"])
    features = extract(crop, mask, method, FEATURES)
    assert 0 < len(features.points) <= FEATURES["max_keypoints"]
    xs = features.points[:, 0]
    assert xs.min() >= np.flatnonzero(mask[128])[0]


def test_sift_auto_retries_on_low_texture_crop():
    smooth = np.full((300, 300, 3), 120, np.uint8)
    cv2.rectangle(smooth, (40, 60), (90, 200), (130, 130, 130), -1)
    crop, mask = normalize_crop(smooth, (20, 20, 100, 250), FEATURES["crop_size"])
    assert extract(crop, mask, "sift_auto", FEATURES).retries > 0


def test_gallery_prefers_large_clean_spaced_boxes():
    obs = [Observation(f, (100, 100, 10 + f, 40), clean=f != 50) for f in range(0, 60)]
    obs.append(Observation(70, (0, 100, 99, 99), clean=True))
    chosen = select_gallery(obs, (640, 360), size=3, min_gap_frames=15, border_px=2)
    assert [o.frame for o in chosen] == [29, 44, 59]


def test_keypoint_similarity_needs_enough_matches():
    rng = np.random.default_rng(1)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8), (3, 3), 0)
    crop, mask = normalize_crop(frame, (20, 20, 60, 220), FEATURES["crop_size"])
    same = extract(crop, mask, "sift", FEATURES)
    other_frame = cv2.GaussianBlur(np.random.default_rng(2).integers(0, 255, (300, 300, 3), dtype=np.uint8), (3, 3), 0)
    other_crop, other_mask = normalize_crop(other_frame, (20, 20, 60, 220), FEATURES["crop_size"])
    other = extract(other_crop, other_mask, "sift", FEATURES)
    assert keypoint_similarity([same], [same], "sift", REID) == pytest.approx(1.0)
    assert keypoint_similarity([same], [other], "sift", REID) == 0.0


def test_geometry_and_combine():
    identity = np.eye(3)
    assert geometry_similarity({1: (10, 10)}, {1: (10, 10)}, identity, 25.0) == pytest.approx(1.0)
    assert geometry_similarity({1: (10, 10)}, {2: (10, 10)}, identity, 25.0) is None
    weights = {"keypoint": 0.4, "mau": 0.4, "san": 0.2}
    assert combine({"keypoint": 1.0, "mau": 0.0, "san": None}, weights) == pytest.approx(0.5)


def test_assign_global_ids_marks_unmatched_tracks():
    rows = assign_global_ids([1, 2], [5, 6], np.array([[0.9, 0.1], [0.1, 0.2]]), threshold=0.3)
    assert ("cam_a", 1, 1) in rows and ("cam_b", 5, 1) in rows
    assert ("cam_a", 2, -1) in rows and ("cam_b", 6, -1) in rows


def _accuracy(out_dir: Path, scene_dir: Path, variant: str) -> int:
    with (out_dir / f"global_ids_{variant}.csv").open(encoding="utf-8") as stream:
        ids = {(r["cam"], int(r["track_id"])): int(r["global_id"]) for r in csv.DictReader(stream)}
    with (scene_dir / "gt_reid.csv").open(encoding="utf-8") as stream:
        pairs = [(int(r["cam_a_id"]), int(r["cam_b_id"])) for r in csv.DictReader(stream)]
    return sum(ids[("cam_a", a)] == ids[("cam_b", b)] != -1 for a, b in pairs)


@pytest.fixture(scope="module")
def s4_result(tmp_path_factory):
    scene_dir = generate("s4", tmp_path_factory.mktemp("s4"), fps=10.0, seed=0)
    out_dir = scene_dir / "reid"
    run(scene_dir, "sift", out_dir, CONFIG)
    return scene_dir, out_dir


def test_identical_clothes_confuse_appearance_but_floor_position_separates_them(s4_result):
    """S4: người 1 và 2 mặc giống hệt nhau. Chỉ ngoại hình thì người 3, 4 vẫn đúng;
    thêm vị trí sàn thì đúng cả 4. Hai kết quả được báo cáo riêng (QĐ-02)."""
    scene_dir, out_dir = s4_result
    assert _accuracy(out_dir, scene_dir, "ngoai_hinh") >= 2
    assert _accuracy(out_dir, scene_dir, "ngoai_hinh_san") == 4
