"""Điểm giống nhau giữa một track ở camera A và một track ở camera B (bước G).

Có ba loại điểm, mỗi loại trong khoảng 0–1:
1. Keypoint (`keypoint_similarity`): với mọi cặp crop trong hai gallery, khớp descriptor bằng
   knnMatch (tìm 2 điểm gần nhất) rồi giữ cặp khớp qua hai bộ lọc:
   - ratio test của Lowe: khoảng cách tốt nhất < ratio × khoảng cách tốt nhì;
   - ràng buộc dọc (XLA 2.8): |y_A / H_A − y_B / H_B| < vertical_tolerance, tức đầu khớp với đầu.
   Có thể bật thêm RANSAC để so sánh (QĐ-06). Lấy số cặp khớp n lớn nhất trên các cặp crop:
       điểm = 0 nếu n < min_matches, ngược lại min(1, n / feature_saturation_matches)
2. Màu: xem src/color_features.py.
3. Vị trí sàn (`geometry_similarity`): ở các frame mà cả hai camera cùng thấy, chiếu điểm chân
   ở A sang ảnh B bằng homography H_AB, đo khoảng cách d tới điểm chân ở B:
       điểm = trung bình exp(−d² / 2σ²)
   Chỉ là thông tin bổ trợ, có thể sai (QĐ-02). Không có frame chung thì trả về None.
Điểm tổng = trung bình có trọng số của các điểm đang có (`combine`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from src.homography import project
from src.keypoint_features import CropFeatures, norm_type


def good_matches(features_a: CropFeatures, features_b: CropFeatures, matcher, cfg: Mapping) -> list:
    """Các cặp keypoint khớp tốt giữa hai crop, sau ratio test, ràng buộc dọc và (tùy chọn) RANSAC."""
    if features_a.descriptors is None or features_b.descriptors is None:
        return []
    if len(features_a.points) < 2 or len(features_b.points) < 2:
        return []
    good = []
    for pair in matcher.knnMatch(features_a.descriptors, features_b.descriptors, k=2):
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance >= float(cfg["ratio_test"]) * second.distance:
            continue
        y_a = features_a.points[best.queryIdx][1] / features_a.height
        y_b = features_b.points[best.trainIdx][1] / features_b.height
        if abs(y_a - y_b) <= float(cfg["vertical_tolerance"]):
            good.append(best)
    if not cfg["use_ransac"] or len(good) < int(cfg["min_matches"]):
        return good
    source = np.float32([features_a.points[m.queryIdx] for m in good])
    target = np.float32([features_b.points[m.trainIdx] for m in good])
    _, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=float(cfg["ransac_threshold_px"]))
    if inliers is None:
        return []
    return [match for match, keep in zip(good, inliers.ravel()) if keep]


def keypoint_similarity(gallery_a: Sequence[CropFeatures], gallery_b: Sequence[CropFeatures], method: str, cfg: Mapping) -> float:
    """Điểm keypoint 0–1 giữa hai gallery: lấy cặp crop có nhiều cặp khớp tốt nhất."""
    matcher = cv2.BFMatcher(norm_type(method))
    best = 0
    for features_a in gallery_a:
        for features_b in gallery_b:
            best = max(best, len(good_matches(features_a, features_b, matcher, cfg)))
    if best < int(cfg["min_matches"]):
        return 0.0
    return min(1.0, best / float(cfg["feature_saturation_matches"]))


def geometry_similarity(feet_a: Mapping[int, Sequence[float]], feet_b: Mapping[int, Sequence[float]], H_ab: np.ndarray, sigma_px: float) -> float | None:
    """Điểm vị trí sàn 0–1 ở các frame cả hai track cùng xuất hiện; None nếu không có frame chung."""
    common = sorted(set(feet_a) & set(feet_b))
    if not common:
        return None
    projected = project(H_ab, [feet_a[f] for f in common])
    targets = np.asarray([feet_b[f] for f in common], dtype=np.float64)
    distances = np.linalg.norm(projected - targets, axis=1)
    return float(np.mean(np.exp(-distances ** 2 / (2.0 * sigma_px ** 2))))


def combine(scores: Mapping[str, float | None], weights: Mapping[str, float]) -> float:
    """Trung bình có trọng số của các điểm khác None: Σ w_i·s_i / Σ w_i."""
    available = {name: value for name, value in scores.items() if value is not None and weights.get(name, 0) > 0}
    total = sum(weights[name] for name in available)
    if total == 0:
        return 0.0
    return float(sum(weights[name] * value for name, value in available.items()) / total)
