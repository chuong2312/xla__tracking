"""Một track: "hồ sơ" của một người trong một camera (bước D).

Vị trí tâm và vận tốc do Kalman giữ (src/kalman.py); kích thước box làm mượt riêng.
Các trạng thái (máy trạng thái của tracker mới):
- tentative: mới thấy, chưa chắc là người thật.
- confirmed: đã chắc chắn.
- merged:    đang dính chung một blob với người khác; vị trí chỉ theo dự đoán.
- static:    mất detection nhưng điểm Lucas-Kanade cho thấy người đứng yên; không bị tính là mất.
- lost:      mất dấu, track kết thúc. ID không bao giờ được dùng lại.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.kalman import ConstantVelocityKalman

STATES = ("tentative", "confirmed", "merged", "static", "lost")
ACTIVE_STATES = ("confirmed", "merged", "static")


@dataclass
class Track:
    """Hồ sơ một người; các bộ đếm phục vụ máy trạng thái."""

    id: int
    kalman: ConstantVelocityKalman
    size: tuple[float, float]
    first_center: tuple[float, float]
    state: str = "tentative"
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 2), dtype=np.float32))
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    merged_frames: int = 0
    static_frames: int = 0
    frames_since_merge: int = 10**6
    age: int = 1
    last_shift: tuple[float, float] | None = None
    template: np.ndarray | None = None
    colors: dict | None = None

    @property
    def center(self) -> tuple[float, float]:
        """Tâm hiện tại (ước lượng của Kalman)."""
        return self.kalman.position

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Box (x, y, w, h) dựng từ tâm và kích thước."""
        cx, cy = self.center
        w, h = self.size
        return (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(w)), int(round(h)))

    @property
    def diagonal(self) -> float:
        """Đường chéo box, dùng để quy đổi khoảng cách theo cỡ người."""
        return float(np.hypot(*self.size))

    def moved_from_start(self) -> float:
        """Khoảng cách (pixel) từ chỗ xuất hiện đầu tiên tới chỗ hiện tại."""
        return float(np.hypot(self.center[0] - self.first_center[0], self.center[1] - self.first_center[1]))
