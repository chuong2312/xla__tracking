"""Đặc trưng màu cho nhận dạng lại (bước F).

1. Chuẩn hóa màu theo camera bằng gray-world (XLA 2.12). Giả định: trung bình màu của
   cả cảnh là màu xám. Với mỗi kênh c (xanh dương, xanh lá, đỏ):
       gain_c = trung bình của cả 3 kênh / trung bình của kênh c
   Nhân ảnh với gain thì hai điện thoại cân bằng trắng khác nhau sẽ cho màu gần nhau hơn.
2. Histogram H-S trong không gian HSV, tính riêng cho thân trên và thân dưới (bỏ phần đầu,
   vì đầu ai cũng giống nhau). H là "màu gì", S là "đậm hay nhạt". Bỏ V (sáng/tối) để bớt
   phụ thuộc ánh sáng. Histogram được chuẩn hóa để tổng bằng 1.
3. So hai histogram bằng khoảng cách Bhattacharyya: 0 là giống hệt, 1 là khác hẳn.
   Độ giống màu = 1 − trung bình khoảng cách của thân trên và thân dưới.
Màu chỉ là đặc trưng bổ trợ: hai người mặc cùng màu thì màu không phân biệt được (kịch bản S4).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

BODY_PARTS = ("upper", "lower")


def gray_world_gains(image_bgr: np.ndarray) -> np.ndarray:
    """Hệ số nhân cho từng kênh màu để trung bình cả ảnh thành màu xám."""
    means = image_bgr.reshape(-1, 3).astype(np.float64).mean(axis=0)
    if np.any(means <= 0):
        return np.ones(3)
    return means.mean() / means


def apply_gains(image_bgr: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    """Nhân từng kênh màu với hệ số tương ứng, giữ giá trị trong khoảng 0–255."""
    scaled = image_bgr.astype(np.float32) * np.asarray(gains, dtype=np.float32)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _band_mask(mask: np.ndarray, band: Sequence[float]) -> np.ndarray:
    """Giữ phần mask nằm trong dải chiều cao [band[0], band[1]) (tỉ lệ 0–1 của crop)."""
    height = mask.shape[0]
    top, bottom = int(round(band[0] * height)), int(round(band[1] * height))
    part = np.zeros_like(mask)
    part[top:bottom] = mask[top:bottom]
    return part


def body_histograms(crop_bgr: np.ndarray, mask: np.ndarray | None, cfg: Mapping) -> dict[str, np.ndarray] | None:
    """Histogram H-S của thân trên và thân dưới; None nếu một phần không còn pixel nào."""
    if mask is None:
        mask = np.full(crop_bgr.shape[:2], 255, dtype=np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    bins = [int(v) for v in cfg["hist_bins"]]
    result = {}
    for part, band in (("upper", cfg["upper_body"]), ("lower", cfg["lower_body"])):
        part_mask = _band_mask(mask, band)
        if not np.any(part_mask):
            return None
        hist = cv2.calcHist([hsv], [0, 1], part_mask, bins, [0, 180, 0, 256])
        result[part] = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1)
    return result


def mean_histograms(histograms: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    """Gộp histogram của nhiều crop (gallery) thành một bằng cách lấy trung bình rồi chuẩn hóa lại."""
    usable = [h for h in histograms if h is not None]
    if not usable:
        return None
    merged = {}
    for part in BODY_PARTS:
        mean = np.mean([h[part] for h in usable], axis=0).astype(np.float32)
        merged[part] = cv2.normalize(mean, None, alpha=1.0, norm_type=cv2.NORM_L1)
    return merged


def color_similarity(hists_a: Mapping[str, np.ndarray] | None, hists_b: Mapping[str, np.ndarray] | None) -> float:
    """Độ giống màu trong khoảng 0–1: 1 − trung bình khoảng cách Bhattacharyya của hai phần thân."""
    if hists_a is None or hists_b is None:
        return 0.0
    distances = [cv2.compareHist(hists_a[p], hists_b[p], cv2.HISTCMP_BHATTACHARYYA) for p in BODY_PARTS]
    return float(1.0 - np.clip(np.mean(distances), 0.0, 1.0))
