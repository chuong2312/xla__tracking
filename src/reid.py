"""Nhận dạng lại giữa hai camera (bước G): mô tả từng track, tính ma trận giống nhau, gán ID toàn cục.

Chuỗi xử lý:
1. Với mỗi track: chọn gallery (src/gallery.py), cắt crop chuẩn, chuẩn hóa màu theo
   camera (gray-world), trích keypoint và histogram màu (bước F).
2. Với mọi cặp (track ở A, track ở B): tính điểm keypoint, màu, vị trí sàn
   (src/reid_similarity.py), rồi gộp thành hai bảng:
   - "ngoai_hinh": chỉ dùng keypoint + màu;
   - "ngoai_hinh_san": thêm vị trí sàn.
   Hai bảng được báo cáo RIÊNG, để thấy phần nhận dạng bằng đặc trưng ảnh làm được đến đâu (QĐ-02).
3. Ghép một-một bằng Hungarian trên chi phí 1 − điểm; cặp có điểm < match_threshold bị loại,
   track không ghép được nhận global_id = −1 ("không khớp ai", XLA 3.2).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.assignment import assign
from src.color_features import apply_gains, body_histograms, color_similarity, mean_histograms
from src.utils import foot_point
from src.gallery import Observation, select_gallery
from src.keypoint_features import CropFeatures, extract, normalize_crop
from src.reid_similarity import combine, geometry_similarity, keypoint_similarity

VARIANTS = ("ngoai_hinh", "ngoai_hinh_san")


@dataclass
class TrackAppearance:
    """Mô tả một track: keypoint của từng crop trong gallery, màu gộp, và điểm chân theo frame."""

    track_id: int
    keypoints: list[CropFeatures]
    colors: dict | None
    feet: dict[int, tuple[float, float]]
    gallery: list[Observation] = field(default_factory=list)


def read_frames(video: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Đọc video một lượt từ đầu, giữ lại các frame cần dùng."""
    capture = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    index = 0
    last = max(wanted, default=-1)
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
        index += 1
    capture.release()
    return frames


def describe_tracks(video: str, observations: Mapping[int, Sequence[Observation]], image_size: Sequence[int],
                    gains: Sequence[float], method: str, features_cfg: Mapping) -> dict[int, TrackAppearance]:
    """Mô tả mọi track của một camera."""
    galleries = {
        track_id: select_gallery(obs, image_size, int(features_cfg["gallery_size"]),
                                 int(features_cfg["gallery_min_gap_frames"]), int(features_cfg["gallery_border_px"]))
        for track_id, obs in observations.items()
    }
    frames = read_frames(video, {o.frame for gallery in galleries.values() for o in gallery})
    result = {}
    for track_id, gallery in galleries.items():
        keypoints, histograms = [], []
        for observation in gallery:
            frame = apply_gains(frames[observation.frame], gains)
            crop, mask = normalize_crop(frame, observation.box, features_cfg["crop_size"])
            keypoints.append(extract(crop, mask, method, features_cfg))
            histograms.append(body_histograms(crop, mask, features_cfg))
        feet = {o.frame: foot_point(o.box) for o in observations[track_id] if o.clean}
        result[track_id] = TrackAppearance(track_id, keypoints, mean_histograms(histograms), feet, gallery)
    return result


def similarity_tables(tracks_a: Mapping[int, TrackAppearance], tracks_b: Mapping[int, TrackAppearance],
                      H_ab: np.ndarray | None, method: str, reid_cfg: Mapping) -> dict[str, np.ndarray]:
    """Các bảng điểm (hàng = track A sắp tăng dần, cột = track B): keypoint, mau, san và hai bảng gộp."""
    ids_a, ids_b = sorted(tracks_a), sorted(tracks_b)
    names = ("keypoint", "mau", "san", *VARIANTS)
    tables = {name: np.full((len(ids_a), len(ids_b)), np.nan) for name in names}
    weights = {"keypoint": reid_cfg["weight_feature"], "mau": reid_cfg["weight_color"], "san": reid_cfg["weight_geometry"]}
    for i, id_a in enumerate(ids_a):
        for j, id_b in enumerate(ids_b):
            a, b = tracks_a[id_a], tracks_b[id_b]
            scores = {
                "keypoint": keypoint_similarity(a.keypoints, b.keypoints, method, reid_cfg),
                "mau": color_similarity(a.colors, b.colors),
                "san": None if H_ab is None else geometry_similarity(a.feet, b.feet, H_ab, float(reid_cfg["geo_sigma_px"])),
            }
            for name, value in scores.items():
                tables[name][i, j] = np.nan if value is None else value
            tables["ngoai_hinh"][i, j] = combine({**scores, "san": None}, weights)
            tables["ngoai_hinh_san"][i, j] = combine(scores, weights)
    return tables


def assign_global_ids(ids_a: Sequence[int], ids_b: Sequence[int], similarity: np.ndarray, threshold: float) -> list[tuple[str, int, int]]:
    """Ghép một-một bằng Hungarian trên 1 − điểm; trả về các dòng (cam, track_id, global_id)."""
    pairs = assign(1.0 - np.nan_to_num(similarity, nan=0.0), max_cost=1.0 - threshold)
    rows = []
    matched_a, matched_b = {}, {}
    for global_id, (i, j) in enumerate(pairs, start=1):
        matched_a[ids_a[i]] = global_id
        matched_b[ids_b[j]] = global_id
    rows += [("cam_a", track_id, matched_a.get(track_id, -1)) for track_id in ids_a]
    rows += [("cam_b", track_id, matched_b.get(track_id, -1)) for track_id in ids_b]
    return rows
