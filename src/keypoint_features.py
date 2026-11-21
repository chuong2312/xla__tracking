"""Keypoint và descriptor cho nhận dạng lại (bước F): SIFT, ORB, sift_auto, ASIFT (QĐ-14).

- Crop người được đưa về một cỡ cố định (mặc định 128×256), giữ tỉ lệ, phần thừa đệm
  màu đen. Một mask đi kèm đánh dấu phần nào là người thật.
- Mask (đã co vào trong vài pixel) được truyền vào tham số `mask` của detector, KHÔNG tô
  đen nền lên ảnh. Tô đen sẽ tạo cạnh giả dọc viền người và sinh keypoint giả (XLA 2.6).
- Mọi phương pháp chỉ giữ tối đa N keypoint mạnh nhất (theo `response`), để khi so sánh
  ORB, SIFT, ASIFT thì khác biệt đến từ thuật toán chứ không phải từ số điểm (XLA 2.7).
- sift_auto: nếu crop có quá ít keypoint thì hạ `contrastThreshold` rồi chạy lại.
- ASIFT: mô phỏng nhiều góc nghiêng rồi chạy SIFT trên từng ảnh (`cv2.AffineFeature_create`);
  không phải phiên bản OpenCV nào cũng có, nhóm đã kiểm chứng trên 5.1.0.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

METHODS = ("sift", "orb", "sift_auto", "asift")


@dataclass
class CropFeatures:
    """Keypoint của một crop: tọa độ (x, y) trong crop chuẩn, descriptor và chiều cao crop."""

    points: np.ndarray
    descriptors: np.ndarray | None
    height: int
    retries: int = 0


def normalize_crop(frame_bgr: np.ndarray, box: Sequence[int], size: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Cắt box ra khỏi frame và đưa về cỡ `size` (rộng, cao), giữ tỉ lệ, đặt giữa nền đen.

    Trả về (crop, mask). Mask = 255 ở phần người thật, 0 ở phần đệm.
    """
    width, height = int(size[0]), int(size[1])
    x, y, w, h = (int(v) for v in box)
    piece = frame_bgr[max(0, y):y + h, max(0, x):x + w]
    if piece.size == 0:
        raise ValueError(f"Box nằm ngoài ảnh: {box}")
    scale = min(width / piece.shape[1], height / piece.shape[0])
    new_w = max(1, int(round(piece.shape[1] * scale)))
    new_h = max(1, int(round(piece.shape[0] * scale)))
    resized = cv2.resize(piece, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    crop[top:top + new_h, left:left + new_w] = resized
    mask[top:top + new_h, left:left + new_w] = 255
    return crop, mask


def norm_type(method: str) -> int:
    """Cách đo khoảng cách descriptor: ORB là chuỗi bit (Hamming), các loại SIFT là số thực (L2)."""
    return cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2


def _detector(method: str, cfg: Mapping, contrast: float | None = None):
    limit = int(cfg["max_keypoints"])
    if method == "orb":
        return cv2.ORB_create(nfeatures=limit, edgeThreshold=int(cfg["orb_edge_threshold"]), patchSize=int(cfg["orb_patch_size"]))
    if method == "asift":
        return cv2.AffineFeature_create(cv2.SIFT_create())
    if contrast is None:
        return cv2.SIFT_create(nfeatures=limit)
    return cv2.SIFT_create(nfeatures=limit, contrastThreshold=float(contrast))


def _strongest(keypoints, descriptors, limit: int):
    """Giữ `limit` keypoint có response lớn nhất (cùng descriptor tương ứng)."""
    if descriptors is None or len(keypoints) <= limit:
        return keypoints, descriptors
    order = np.argsort([-kp.response for kp in keypoints])[:limit]
    return [keypoints[i] for i in order], descriptors[order]


def extract(crop_bgr: np.ndarray, mask: np.ndarray, method: str, cfg: Mapping) -> CropFeatures:
    """Tìm keypoint và descriptor trên crop chuẩn theo phương pháp `method`."""
    if method not in METHODS:
        raise ValueError(f"features.method phải là một trong {METHODS}")
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    erode = int(cfg["mask_erode_px"])
    if erode > 0:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode + 1, 2 * erode + 1)))
    retries = 0
    if method == "sift_auto":
        thresholds = [float(v) for v in cfg["sift_auto_contrast_thresholds"]]
        for retries, contrast in enumerate(thresholds):
            keypoints, descriptors = _detector("sift", cfg, contrast).detectAndCompute(gray, mask)
            if len(keypoints) >= int(cfg["sift_auto_min_keypoints"]):
                break
    else:
        keypoints, descriptors = _detector(method, cfg).detectAndCompute(gray, mask)
    keypoints, descriptors = _strongest(keypoints, descriptors, int(cfg["max_keypoints"]))
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    return CropFeatures(points, descriptors, crop_bgr.shape[0], retries)
