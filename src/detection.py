"""Tìm vùng có người trong một frame (bước C).

Thứ tự các bước:
1. Mô hình nền cho mask foreground (src/background.py).
2. Che ROI và vùng bỏ qua NGAY sau bước 1, trước morphology, để phép giãn không lan
   vào vùng đã che (XLA 2.16).
3. Opening: xóa chấm nhiễu nhỏ.
4. Closing với kernel dựng đứng (mặc định rộng 3, cao 9): nối thân với chân mà ít nối
   ngang hai người đứng cạnh nhau.
5. Connected components: mỗi cụm pixel liền nhau cho ra một box.
6. Lọc box theo diện tích và tỉ lệ cao/rộng.
Sau khi tracker cập nhật, gọi `update_background` với box của những người đang theo dõi,
để nền không cập nhật ở chỗ có người.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from src.background import make_background_model

Box = tuple[int, int, int, int]


def _kernel(size: Sequence[int]) -> np.ndarray:
    """Kernel hình chữ nhật cỡ (rộng, cao)."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (int(size[0]), int(size[1])))


class PersonDetector:
    """Trả về box các vùng chuyển động có kích thước và hình dạng giống người."""

    def __init__(self, cfg: Mapping, roi: Sequence[int] | None = None, ignore_regions: Sequence[Sequence[int]] = ()):
        self._cfg = cfg
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
        boxes: list[Box] = []
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
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
