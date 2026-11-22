"""Tìm vùng có người trong một frame (bước C).

Thứ tự các bước:
1. Mô hình nền cho mask foreground (src/detect/background.py).
2. Che ROI và vùng bỏ qua NGAY sau bước 1, trước morphology, để phép giãn không lan
   vào vùng đã che (XLA 2.16).
3. Opening: xóa chấm nhiễu nhỏ.
4. Closing với kernel dựng đứng (mặc định rộng 3, cao 9): nối thân với chân mà ít nối
   ngang hai người đứng cạnh nhau.
5. Connected components: mỗi cụm pixel liền nhau cho ra một mảnh.
6. Gộp các mảnh nằm thẳng hàng theo chiều dọc (đầu, chân của cùng một người bị đứt vì áo trùng
   màu nền), CHỈ khi có mô hình chiều cao (src/detect/height_model.py) và box gộp không cao quá
   `group_max_height_ratio` × chiều cao người mong đợi; nếu không, hai người đứng trước – sau
   trong hành lang sẽ bị gộp nhầm (QĐ-26).
7. Lọc box theo diện tích và tỉ lệ cao/rộng.
Sau khi tracker cập nhật, gọi `update_background` với box của những người đang theo dõi,
để nền không cập nhật ở chỗ có người.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from src.detect.background import make_background_model
from src.detect.height_model import HeightModel

Box = tuple[int, int, int, int]


def _kernel(size: Sequence[int]) -> np.ndarray:
    """Kernel hình chữ nhật cỡ (rộng, cao)."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (int(size[0]), int(size[1])))


def _stacked(a: Box, b: Box, max_gap_ratio: float) -> bool:
    """Hai mảnh nằm thẳng hàng theo chiều dọc và cách nhau không xa.

    - Phần chung theo chiều ngang ≥ 50% bề rộng mảnh hẹp hơn (một trên, một dưới).
    - Khoảng hở theo chiều dọc ≤ max_gap_ratio × chiều cao mảnh cao hơn.
    """
    overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    if overlap < 0.5 * min(a[2], b[2]):
        return False
    gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
    return gap <= max_gap_ratio * max(a[3], b[3])


def _union(a: Box, b: Box) -> Box:
    """Box nhỏ nhất bao cả hai box."""
    left, top = min(a[0], b[0]), min(a[1], b[1])
    right, bottom = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (left, top, right - left, bottom - top)


def group_vertical_pieces(pieces: Sequence[Box], max_gap_ratio: float, heights: HeightModel, max_height_ratio: float) -> list[Box]:
    """Gộp các mảnh của cùng một người (đầu – thân – chân bị đứt) thành một box.

    Hai mảnh a, b được gộp khi chúng thẳng hàng theo chiều dọc (_stacked) VÀ
        chiều cao box gộp ≤ max_height_ratio × h(y_chân của box gộp)
    với h là chiều cao người mong đợi. Lặp lại cho đến khi không gộp thêm được.
    """
    boxes = [tuple(p) for p in pieces]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                joined = _union(boxes[i], boxes[j])
                tall_enough = joined[3] <= max_height_ratio * heights.expected(joined[1] + joined[3])
                if tall_enough and _stacked(boxes[i], boxes[j], max_gap_ratio):
                    boxes[i] = joined
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


class PersonDetector:
    """Trả về box các vùng chuyển động có kích thước và hình dạng giống người."""

    def __init__(self, cfg: Mapping, roi: Sequence[int] | None = None, ignore_regions: Sequence[Sequence[int]] = (),
                 heights: HeightModel | None = None):
        self._cfg = cfg
        self.heights = heights
        self._model = make_background_model(cfg)
        self._roi = roi
        self._ignore = list(ignore_regions)
        self.last_mask: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        """Mô hình nền đã sẵn sàng chưa (median_running cần gom đủ frame đầu)."""
        return self._model.ready

    def _apply_regions(self, mask: np.ndarray) -> np.ndarray:
        """Đưa về 0 các pixel ngoài ROI [x1, y1, x2, y2] và trong các vùng bỏ qua [x, y, w, h]."""
        if self._roi is not None:
            x1, y1, x2, y2 = (int(v) for v in self._roi)
            keep = np.zeros_like(mask)
            keep[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
            mask = cv2.bitwise_and(mask, keep)
        for x, y, w, h in self._ignore:
            mask[max(0, int(y)):max(0, int(y + h)), max(0, int(x)):max(0, int(x + w))] = 0
        return mask

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        """Tìm box các vùng giống người; mask sau morphology lưu ở `last_mask`."""
        cfg = self._cfg
        mask = self._apply_regions(self._model.apply(frame_bgr))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(cfg["open_kernel"]), iterations=int(cfg["open_iterations"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(cfg["close_kernel"]), iterations=int(cfg["close_iterations"]))
        self.last_mask = mask
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        pieces = [tuple(int(v) for v in stats[i][:4]) for i in range(1, count) if stats[i][4] >= int(cfg["min_piece_area"])]
        if self.heights is not None:
            pieces = group_vertical_pieces(pieces, float(cfg["group_max_gap_ratio"]), self.heights, float(cfg["group_max_height_ratio"]))
        boxes: list[Box] = []
        for x, y, w, h in pieces:
            area = w * h
            ratio = h / w if w else float("inf")
            if float(cfg["min_area"]) <= area <= float(cfg["max_area"]) and float(cfg["min_ratio"]) <= ratio <= float(cfg["max_ratio"]):
                boxes.append((x, y, w, h))
        return boxes

    def update_background(self, frame_bgr: np.ndarray, people_boxes: Sequence[Sequence[float]]) -> None:
        """Cập nhật nền, trừ vùng quanh các box người (nới thêm exclude_margin_px mỗi phía)."""
        exclude = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        margin = int(self._cfg["exclude_margin_px"])
        for x, y, w, h in people_boxes:
            left, top = max(0, int(x) - margin), max(0, int(y) - margin)
            exclude[top:int(y + h) + margin, left:int(x + w) + margin] = 255
        self._model.update(frame_bgr, exclude)
