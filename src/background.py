"""Mô hình nền để tìm vùng chuyển động (bước C). Có hai cách, chọn bằng `detection.background` (QĐ-03).

1. "median_running" (thuần xử lý ảnh):
   - Khởi tạo: ảnh nền B = trung vị theo thời gian của `init_frames` frame đầu (cảnh trống).
   - (Tùy chọn) Bù độ sáng: chia frame cho g = trung vị(sáng frame / sáng nền), vì camera hay
     tự chỉnh độ sáng cả ảnh khi người mặc đồ tối đi vào.
   - Foreground: pixel có ‖I − B‖ > diff_threshold (độ dài vector chênh lệch 3 kênh màu).
   - Khử bóng (Cucchiara 2003): pixel foreground là bóng nếu trong HSV:
       v_min ≤ V_I / V_B ≤ v_max,  |S_I − S_B| ≤ s_max,  |H_I − H_B| ≤ h_max
     tức tối đi nhưng gần như giữ nguyên màu.
   - Cập nhật CÓ CHỌN LỌC: B ← (1 − a)·B + a·I chỉ ở pixel KHÔNG thuộc người đang được
     theo dõi (cv2.accumulateWeighted với mask). Người đứng yên vì thế không bị "nuốt"
     vào nền (XLA 2.2), còn ánh sáng đổi dần thì nền vẫn theo kịp.
2. "mog2": mô hình nền hỗn hợp Gaussian thích nghi của OpenCV. Chỉ phát hiện foreground,
   không biết đâu là người. learningRate áp cho cả ảnh nên không cập nhật có chọn lọc được.
Cả hai đều trả về mask 0/255 cùng cỡ ảnh.
"""
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def shadow_mask(frame_bgr: np.ndarray, background_bgr: np.ndarray, cfg: Mapping) -> np.ndarray:
    """Mảng bool: pixel nào trông như bóng (tối đi nhưng giữ màu) so với nền."""
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_back = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ratio = hsv_frame[..., 2] / np.maximum(hsv_back[..., 2], 1.0)
    hue_diff = np.abs(hsv_frame[..., 0] - hsv_back[..., 0])
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
    sat_diff = np.abs(hsv_frame[..., 1] - hsv_back[..., 1])
    return (
        (ratio >= float(cfg["shadow_v_min"])) & (ratio <= float(cfg["shadow_v_max"]))
        & (sat_diff <= float(cfg["shadow_s_max"])) & (hue_diff <= float(cfg["shadow_h_max"]))
    )


class MedianRunningBackground:
    """Nền trung vị + cập nhật có chọn lọc."""

    def __init__(self, cfg: Mapping):
        self._cfg = cfg
        self._init_frames = int(cfg["init_frames"])
        self._threshold = float(cfg["diff_threshold"])
        self._rate = float(cfg["update_rate"])
        self._buffer: list[np.ndarray] = []
        self.background: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        """Đã có ảnh nền hay còn đang gom các frame đầu."""
        return self.background is not None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Trả về mask foreground; trong lúc gom frame đầu thì trả mask rỗng."""
        if self.background is None:
            self._buffer.append(frame_bgr)
            if len(self._buffer) >= self._init_frames:
                self.background = np.median(np.stack(self._buffer), axis=0).astype(np.float32)
                self._buffer.clear()
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        frame = frame_bgr.astype(np.float32)
        if self._cfg.get("exposure_compensation", False):
            frame = frame / self._exposure_gain(frame)
        difference = np.linalg.norm(frame - self.background, axis=2)
        foreground = difference > self._threshold
        foreground &= ~shadow_mask(frame_bgr, self.background.astype(np.uint8), self._cfg)
        return foreground.astype(np.uint8) * 255

    def _exposure_gain(self, frame: np.ndarray) -> float:
        """Tỉ lệ sáng chung giữa frame và nền: g = trung vị(độ sáng frame / độ sáng nền).

        Camera điện thoại / camera an ninh thường tự chỉnh độ sáng khi có người mặc đồ tối đi
        vào, làm CẢ ảnh sáng hoặc tối đi. Người chỉ chiếm một phần nhỏ của ảnh, nên trung vị
        của tỉ lệ phản ánh độ sáng chung; chia frame cho g trước khi so với nền thì phần nền
        không còn bị tưởng là chuyển động.
        """
        bright = frame.mean(axis=2)
        reference = self.background.mean(axis=2)
        valid = reference > 10
        if not np.any(valid):
            return 1.0
        return float(np.clip(np.median(bright[valid] / reference[valid]), 0.5, 2.0))

    def update(self, frame_bgr: np.ndarray, exclude_mask: np.ndarray) -> None:
        """Cập nhật nền ở những pixel không bị loại trừ (không thuộc người đang theo dõi)."""
        if self.background is None:
            return
        allowed = (exclude_mask == 0).astype(np.uint8)
        cv2.accumulateWeighted(frame_bgr, self.background, self._rate, mask=allowed)


class Mog2Background:
    """MOG2 của OpenCV; dùng để so sánh với median_running."""

    def __init__(self, cfg: Mapping):
        self._model = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg["mog2_history"]), varThreshold=float(cfg["mog2_var_threshold"]), detectShadows=True
        )
        self._rate = float(cfg["mog2_learning_rate"])

    @property
    def ready(self) -> bool:
        """MOG2 luôn trả kết quả, kể cả ở frame đầu."""
        return True

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Giữ pixel bằng 255 (foreground), bỏ pixel 127 (bóng do MOG2 tự đánh dấu)."""
        raw = self._model.apply(frame_bgr, learningRate=self._rate)
        return np.where(raw == 255, 255, 0).astype(np.uint8)

    def update(self, frame_bgr: np.ndarray, exclude_mask: np.ndarray) -> None:
        """MOG2 tự cập nhật trong `apply`, không loại trừ được vùng nào."""


def make_background_model(cfg: Mapping):
    """Tạo mô hình nền theo `detection.background`."""
    kinds = {"median_running": MedianRunningBackground, "mog2": Mog2Background}
    if cfg["background"] not in kinds:
        raise ValueError(f"detection.background phải là một trong {sorted(kinds)}")
    return kinds[cfg["background"]](cfg)
