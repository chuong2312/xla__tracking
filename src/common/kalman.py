"""Bộ lọc Kalman vận tốc hằng cho tâm của một người (bước D). Thầy cho dùng (QĐ-05).

Trạng thái x = [cx, cy, vx, vy]: vị trí tâm (pixel) và vận tốc (pixel/frame).
Mỗi frame có hai bước:
1. Dự đoán (predict), với Δt = 1 frame:
       x ← F·x,     F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
   tức "vị trí mới = vị trí cũ + vận tốc". Độ không chắc chắn tăng: P ← F·P·Fᵀ + Q.
2. Hiệu chỉnh (correct) khi đo được vị trí z = [cx, cy]:
       K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹,   x ← x + K·(z − H·x),   P ← (I − K·H)·P
   Kết quả là trung bình có trọng số giữa dự đoán và phép đo: nguồn nào ít nhiễu hơn
   (R hoặc P nhỏ hơn) thì được tin nhiều hơn.
Khi người bị che, không có phép đo: chỉ predict, nên vị trí tiếp tục đi theo vận tốc cũ.
Các mức nhiễu do nhóm đặt trong section `kalman` của config, không học từ dữ liệu.
"""
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

SOURCES = ("detection", "lk")


def _process_noise(acceleration_std: float) -> np.ndarray:
    """Ma trận Q cho mô hình "gia tốc ngẫu nhiên" với Δt = 1 frame.

    Gia tốc ngẫu nhiên a (độ lệch chuẩn σ) làm vị trí lệch a/2 và vận tốc lệch a sau
    một frame, nên với mỗi trục: Q = σ² · [[1/4, 1/2], [1/2, 1]] (vị trí, vận tốc).
    """
    variance = acceleration_std ** 2
    q = np.zeros((4, 4), dtype=np.float32)
    for pos, vel in ((0, 2), (1, 3)):
        q[pos, pos] = variance / 4.0
        q[pos, vel] = variance / 2.0
        q[vel, pos] = variance / 2.0
        q[vel, vel] = variance
    return q


class ConstantVelocityKalman:
    """Kalman cho tâm một người; dùng cv2.KalmanFilter với ma trận do nhóm tự đặt."""

    def __init__(self, cx: float, cy: float, cfg: Mapping):
        self._noise_std = {
            "detection": float(cfg["detection_noise_px"]),
            "lk": float(cfg["lk_noise_px"]),
        }
        initial_velocity_std = float(cfg["initial_velocity_std_px"])
        self._filter = cv2.KalmanFilter(4, 2)
        self._filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self._filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self._filter.processNoiseCov = _process_noise(float(cfg["acceleration_std_px"]))
        position_var = self._noise_std["detection"] ** 2
        velocity_var = initial_velocity_std ** 2
        self._filter.errorCovPost = np.diag([position_var, position_var, velocity_var, velocity_var]).astype(np.float32)
        self._filter.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)

    def predict(self) -> tuple[float, float]:
        """Bước dự đoán; trả về vị trí tâm dự đoán cho frame mới."""
        state = self._filter.predict()
        return (float(state[0, 0]), float(state[1, 0]))

    def correct(self, cx: float, cy: float, source: str = "detection") -> tuple[float, float]:
        """Bước hiệu chỉnh bằng một phép đo; `source` chọn mức nhiễu đo (detection hoặc lk)."""
        if source not in SOURCES:
            raise ValueError(f"source phải là một trong {SOURCES}")
        variance = self._noise_std[source] ** 2
        self._filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * variance
        state = self._filter.correct(np.array([[cx], [cy]], dtype=np.float32))
        return (float(state[0, 0]), float(state[1, 0]))

    @property
    def position(self) -> tuple[float, float]:
        """Vị trí tâm ước lượng hiện tại."""
        state = self._filter.statePost
        return (float(state[0, 0]), float(state[1, 0]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Vận tốc ước lượng hiện tại (pixel/frame)."""
        state = self._filter.statePost
        return (float(state[2, 0]), float(state[3, 0]))
