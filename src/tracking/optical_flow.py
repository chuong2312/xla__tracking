"""Đo chuyển động của một người bằng Lucas-Kanade (bước D).

1. Chọn điểm: góc tốt (`goodFeaturesToTrack`) chỉ trong vùng box ∩ foreground đã co lại
   (XLA 2.1), trừ đi box của những người khác. Nhờ vậy không lấy điểm trên nền: điểm nền
   có độ dịch bằng 0 sẽ kéo box "dính" vào nền.
2. Bám điểm sang frame mới bằng LK nhiều tầng (pyramid), rồi kiểm tra đi-rồi-về
   (forward-backward): chạy ngược lại, điểm nào không quay về gần chỗ cũ trong vòng
   `fb_threshold_px` pixel thì bỏ.
3. Độ dịch của người = TRUNG VỊ độ dịch các điểm còn lại. Trung vị bền với vài điểm sai.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

EMPTY = np.empty((0, 1, 2), dtype=np.float32)


def find_points(gray: np.ndarray, box: Sequence[float], foreground: np.ndarray | None,
                other_boxes: Sequence[Sequence[float]], cfg: Mapping) -> np.ndarray:
    """Tìm điểm góc trong box ∩ foreground (đã co), trừ box của người khác; trả mảng N×1×2."""
    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = (int(round(v)) for v in box)
    mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 255
    if foreground is not None:
        erode = int(cfg["points_mask_erode_px"])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode + 1, 2 * erode + 1))
        mask = cv2.bitwise_and(mask, cv2.erode(foreground, kernel))
    for ox, oy, ow, oh in other_boxes:
        mask[max(0, int(oy)):max(0, int(oy + oh)), max(0, int(ox)):max(0, int(ox + ow))] = 0
    if not np.any(mask):
        return EMPTY
    points = cv2.goodFeaturesToTrack(gray, maxCorners=int(cfg["max_corners"]), qualityLevel=float(cfg["quality_level"]),
                                     minDistance=float(cfg["min_distance_px"]), mask=mask, blockSize=7)
    return EMPTY if points is None else points.astype(np.float32).reshape(-1, 1, 2)


def track_points(previous_gray: np.ndarray, gray: np.ndarray, points: np.ndarray, cfg: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Bám điểm từ frame trước sang frame này; trả về (vị trí cũ, vị trí mới) của các điểm tin cậy, dạng N×2."""
    if len(points) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    params = {"winSize": tuple(int(v) for v in cfg["lk_win_size"]), "maxLevel": int(cfg["lk_max_level"])}
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None, **params)
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, forward, None, **params)
    old = points.reshape(-1, 2)
    new = forward.reshape(-1, 2)
    error = np.linalg.norm(old - backward.reshape(-1, 2), axis=1)
    keep = status.ravel().astype(bool) & status_back.ravel().astype(bool) & (error <= float(cfg["fb_threshold_px"]))
    return old[keep], new[keep]


def median_shift(old: np.ndarray, new: np.ndarray, min_points: int) -> tuple[float, float] | None:
    """Độ dịch trung vị (dx, dy); None nếu còn ít hơn `min_points` điểm tin cậy."""
    if len(old) < min_points:
        return None
    dx, dy = np.median(new - old, axis=0)
    return (float(dx), float(dy))
