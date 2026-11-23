"""Ổn định hình: bù việc camera bị trôi hoặc rung trước khi tìm người (QĐ-30).

Vấn đề: bước tìm người so từng frame với ảnh nền, nên giả định camera đứng yên. Video quay cầm
tay hoặc chân máy không chắc thì cả khung hình trôi dần (S1: tới khoảng 40 px sau 18 giây);
mép cửa sổ, cột, lan can bị coi là "đang chuyển động" và sinh rất nhiều khung báo nhầm.

Cách làm, cho mỗi frame:
1. Frame đầu là frame tham chiếu. Chọn sẵn các điểm góc trên đó (goodFeaturesToTrack).
2. Bám các điểm đó sang frame hiện tại bằng Lucas-Kanade nhiều tầng.
3. Ước lượng phép biến đổi M (dời + xoay + co giãn đều, 4 ẩn) đưa frame hiện tại về frame
   tham chiếu bằng RANSAC (cv2.estimateAffinePartial2D). RANSAC bỏ các điểm nằm trên người đang
   đi, vì chúng không theo cùng một chuyển động với nền.
4. Nếu M làm các góc ảnh dời ít hơn `min_shift_px` thì coi như camera đứng yên, giữ nguyên frame
   (không nội suy làm mờ ảnh). Ngược lại, biến đổi frame bằng M (cv2.warpAffine).
Vùng mép không có dữ liệu sau khi biến đổi bị đánh dấu là không hợp lệ (`valid_mask`) để bước tìm
người bỏ qua. Box tìm được trong tọa độ đã ổn định được đổi ngược về tọa độ video gốc bằng M⁻¹
(`to_original`), nên tracks.csv, đáp án, nhận dạng lại và video debug vẫn dùng tọa độ gốc.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def corner_shift(matrix: np.ndarray, width: int, height: int) -> float:
    """Độ dời lớn nhất (pixel) của 4 góc ảnh dưới phép biến đổi affine 2×3."""
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64)
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    return float(np.max(np.linalg.norm(moved - corners, axis=1)))


class Stabilizer:
    """Đưa mọi frame về tọa độ của frame đầu; nhớ phép biến đổi của frame vừa xử lý."""

    def __init__(self, cfg: Mapping):
        self._cfg = cfg
        self._reference: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self.matrix = IDENTITY.copy()
        self.valid_mask: np.ndarray | None = None

    def _track(self, gray: np.ndarray) -> np.ndarray | None:
        """Phép biến đổi frame hiện tại → frame tham chiếu; None nếu không đủ điểm tin cậy."""
        cfg = self._cfg
        size = tuple(int(v) for v in cfg["lk_win_size"])
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self._reference, gray, self._points, None,
                                                    winSize=size, maxLevel=int(cfg["lk_max_level"]))
        good = status.ravel() == 1
        if good.sum() < int(cfg["min_inliers"]):
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(moved[good], self._points[good], method=cv2.RANSAC,
                                                      ransacReprojThreshold=float(cfg["ransac_threshold_px"]))
        if matrix is None or inliers is None or inliers.sum() < int(cfg["min_inliers"]):
            return None
        return matrix

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Trả về frame đã ổn định; cập nhật `matrix` và `valid_mask` (None = cả ảnh hợp lệ)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if self._reference is None:
            self._reference = gray
            self._points = cv2.goodFeaturesToTrack(gray, int(self._cfg["max_corners"]), float(self._cfg["quality_level"]),
                                                   float(self._cfg["min_distance_px"]))
            return frame_bgr
        matrix = self._track(gray) if self._points is not None else None
        self.matrix = self.matrix if matrix is None else matrix
        if corner_shift(self.matrix, width, height) < float(self._cfg["min_shift_px"]):
            self.valid_mask = None
            return frame_bgr
        warped = cv2.warpAffine(frame_bgr, self.matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full((height, width), 255, np.uint8), self.matrix, (width, height), borderValue=0)
        border = int(self._cfg["border_px"])
        self.valid_mask = cv2.erode(valid, np.ones((2 * border + 1, 2 * border + 1), np.uint8)) if border > 0 else valid
        return warped

    def to_original(self, box: Sequence[float]) -> tuple[int, int, int, int]:
        """Đổi box (x, y, w, h) từ tọa độ đã ổn định về tọa độ video gốc bằng M⁻¹."""
        x, y, w, h = box
        inverse = cv2.invertAffineTransform(self.matrix)
        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float64)
        moved = corners @ inverse[:, :2].T + inverse[:, 2]
        left, top = moved.min(axis=0)
        right, bottom = moved.max(axis=0)
        return int(round(left)), int(round(top)), int(round(right - left)), int(round(bottom - top))
