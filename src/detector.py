"""Phát hiện vùng chuyển động bằng MOG2 và xử lý hình thái học."""
from collections.abc import Mapping, Sequence
from typing import Any
import cv2
import numpy as np

class Detector:
    """Phân đoạn chuyển động và trả các hộp ứng viên cho tracker."""
    def __init__(self, cfg: Any):
        """Khởi tạo MOG2 và nạp toàn bộ tham số từ config."""
        detector_cfg, scene_cfg = _config_sections(cfg)
        self._width = _get(scene_cfg, "resize_width", None)
        self._height = _get(scene_cfg, "resize_height", None)
        self._roi = _get(scene_cfg, "roi", None)
        self._ignore_regions = _get(scene_cfg, "ignore_regions", [])
        self._blur_kernel = _odd_kernel(detector_cfg, "blur_kernel")
        self._history = _positive_int(detector_cfg, "history")
        self._var_threshold = _positive_number(detector_cfg, "varThreshold")
        self._learning_rate = _number(detector_cfg, "learningRate")
        if not 0 <= self._learning_rate <= 1:
            raise ValueError("detector.learningRate phải nằm trong [0, 1]")
        self._open_kernel = _odd_kernel(detector_cfg, "opening_kernel")
        self._close_kernel = _odd_kernel(detector_cfg, "closing_kernel")
        self._dilate_kernel = _odd_kernel(detector_cfg, "dilation_kernel")
        self._open_iterations = _nonnegative_int(detector_cfg, "opening_iterations")
        self._close_iterations = _nonnegative_int(detector_cfg, "closing_iterations")
        self._dilate_iterations = _nonnegative_int(detector_cfg, "dilation_iterations")
        self._min_area = _positive_number(detector_cfg, "min_area")
        self._max_area = _positive_number(detector_cfg, "max_area")
        self._min_ratio = _positive_number(detector_cfg, "min_ratio")
        self._max_ratio = _positive_number(detector_cfg, "max_ratio")
        if self._max_area < self._min_area or self._max_ratio < self._min_ratio:
            raise ValueError("Khoảng diện tích/tỷ lệ detector không hợp lệ")
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=True,
        )
        self._mask: np.ndarray | None = None

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Xử lý frame BGR đã resize và trả box (x,y,w,h)."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr phải là ảnh BGR 3 kênh")
        blurred = cv2.GaussianBlur(frame_bgr, (self._blur_kernel, self._blur_kernel), 0)
        raw_mask = self._mog2.apply(blurred, learningRate=self._learning_rate)
        mask = cv2.compare(raw_mask, 255, cv2.CMP_EQ)
        self._apply_regions(mask)
        if self._open_iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(self._open_kernel), iterations=self._open_iterations)
        if self._close_iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(self._close_kernel), iterations=self._close_iterations)
        if self._dilate_iterations:
            mask = cv2.dilate(mask, _kernel(self._dilate_kernel), iterations=self._dilate_iterations)
        self._apply_regions(mask)
        self._mask = mask
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: list[tuple[int, int, int, int]] = []
        for index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[index])
            ratio = height / width if width else float("inf")
            if self._min_area <= area <= self._max_area and self._min_ratio <= ratio <= self._max_ratio:
                boxes.append((x, y, width, height))
        return boxes

    def get_foreground_mask(self) -> np.ndarray:
        """Trả mask 0/255 đã qua lọc bóng, ROI và ignore regions."""
        if self._mask is None:
            raise RuntimeError("Chưa có foreground mask; gọi detect(frame) trước")
        return self._mask.copy()

    def _apply_regions(self, mask: np.ndarray) -> None:
        """Đưa pixel ngoài ROI và trong vùng che về 0 an toàn."""
        height, width = mask.shape
        if self._roi is not None:
            x1, y1, x2, y2 = (int(v) for v in self._roi)
            roi_mask = np.zeros_like(mask)
            left, top = max(0, x1), max(0, y1)
            right, bottom = min(width, x2), min(height, y2)
            if right > left and bottom > top:
                roi_mask[top:bottom, left:right] = 255
            cv2.bitwise_and(mask, roi_mask, dst=mask)
        for region in self._ignore_regions or []:
            if len(region) != 4:
                raise ValueError("Mỗi scene.ignore_regions cần (x,y,w,h)")
            x, y, box_w, box_h = (int(v) for v in region)
            left, top = max(0, x), max(0, y)
            right, bottom = min(width, x + box_w), min(height, y + box_h)
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 0

def _config_sections(cfg: Any) -> tuple[Any, Any]:
    if isinstance(cfg, Mapping):
        return cfg["detector"], cfg["scene"]
    return cfg.detector, cfg.scene

def _get(section: Any, key: str, default: Any) -> Any:
    return section.get(key, default) if isinstance(section, Mapping) else getattr(section, key, default)

def _positive_int(section: Any, key: str) -> int:
    value = _get(section, key, None)
    if type(value) is not int or value <= 0: raise ValueError(f"detector.{key} phải là số nguyên dương")
    return value

def _nonnegative_int(section: Any, key: str) -> int:
    value = _get(section, key, None)
    if type(value) is not int or value < 0: raise ValueError(f"detector.{key} phải là số nguyên không âm")
    return value

def _positive_number(section: Any, key: str) -> float:
    value = _get(section, key, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0: raise ValueError(f"detector.{key} phải là số dương")
    return float(value)

def _number(section: Any, key: str) -> float:
    value = _get(section, key, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool): raise ValueError(f"detector.{key} phải là số")
    return float(value)

def _odd_kernel(section: Any, key: str) -> int:
    value = _positive_int(section, key)
    if value % 2 == 0: raise ValueError(f"detector.{key} phải là số lẻ")
    return value

def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
