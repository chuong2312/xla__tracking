"""Đếm người đi vào / đi ra qua cửa (bước E). Thầy yêu cầu bắt buộc phần này (QĐ-12).

Ý tưởng:
1. Mỗi track được đại diện bằng điểm chân p = (cx, y + h), tức chỗ chân chạm sàn.
2. Tính khoảng cách có dấu d từ p tới vạch cửa, đi từ P1 = (x1, y1) tới P2 = (x2, y2):
       d = ((x2 − x1)·(py − y1) − (y2 − y1)·(px − x1)) / ‖P2 − P1‖
   Trên ảnh (trục y hướng xuống), d > 0 nghĩa là p nằm bên phải vạch khi nhìn
   theo hướng P1 → P2; d < 0 là bên trái.
3. Vùng đệm (hysteresis): khi |d| ≤ margin thì coi như người đang đứng trên vạch,
   chưa quyết định ở phía nào. Nhờ vậy người đứng rung trên vạch không bị đếm nhiều lần.
4. Khi phía đã xác định của một track đổi dấu, và chân nằm trong bề rộng cửa,
   thì phát một sự kiện: sang phía "bên trong" là IN, sang phía còn lại là OUT.
Chỉ dùng vị trí điểm chân, không dùng màu sắc hay ngoại hình.
Sự kiện được phát ngay trong `update`, nên tracker phải gọi `update` TRƯỚC khi
kết thúc một track (XLA 2.17).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Line = tuple[float, float, float, float]
Point = tuple[float, float]
INSIDE_SIDES = ("positive", "negative")


@dataclass(frozen=True)
class DoorEvent:
    """Một lượt đi qua cửa."""

    frame: int
    track_id: int
    direction: str


def foot_point(box: Sequence[float]) -> Point:
    """Điểm chân của box (x, y, w, h): điểm giữa cạnh dưới."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)


def signed_distance(line: Sequence[float], point: Sequence[float]) -> float:
    """Khoảng cách có dấu (pixel) từ điểm tới đường thẳng chứa vạch cửa; công thức ở đầu file."""
    x1, y1, x2, y2 = line
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        raise ValueError("Vạch cửa phải có hai điểm đầu mút khác nhau")
    px, py = point
    return ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / length


def within_door(line: Sequence[float], point: Sequence[float]) -> bool:
    """Hình chiếu vuông góc của điểm lên vạch có nằm giữa hai đầu vạch không.

    t = ((p − P1) · (P2 − P1)) / ‖P2 − P1‖². Điểm nằm trong bề rộng cửa khi 0 ≤ t ≤ 1.
    """
    x1, y1, x2, y2 = line
    px, py = point
    length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / length_sq
    return 0.0 <= t <= 1.0


class DoorCounter:
    """Theo dõi mỗi track đang ở phía nào của vạch cửa và phát sự kiện IN/OUT."""

    def __init__(self, line: Sequence[float], margin_px: float, inside_side: str):
        if len(line) != 4:
            raise ValueError("Vạch cửa cần 4 số x1, y1, x2, y2")
        if margin_px < 0:
            raise ValueError("events.crossing_margin phải không âm")
        if inside_side not in INSIDE_SIDES:
            raise ValueError("events.inside_side phải là 'positive' hoặc 'negative'")
        self._line: Line = tuple(float(v) for v in line)
        self._margin = float(margin_px)
        self._inside_sign = 1 if inside_side == "positive" else -1
        self._sides: dict[int, int] = {}
        self.events: list[DoorEvent] = []

    @classmethod
    def from_config(cls, events_cfg: Mapping, line: Sequence[float]) -> "DoorCounter":
        """Tạo bộ đếm từ section `events` của config và vạch cửa của camera."""
        return cls(line, events_cfg["crossing_margin"], events_cfg["inside_side"])

    def update(self, frame: int, track_id: int, foot: Sequence[float]) -> DoorEvent | None:
        """Cập nhật vị trí chân của một track; trả về sự kiện nếu track vừa đi qua cửa."""
        distance = signed_distance(self._line, foot)
        if abs(distance) <= self._margin:
            return None
        side = 1 if distance > 0 else -1
        previous = self._sides.get(track_id)
        self._sides[track_id] = side
        if previous is None or previous == side:
            return None
        if not within_door(self._line, foot):
            return None
        direction = "IN" if side == self._inside_sign else "OUT"
        event = DoorEvent(frame, track_id, direction)
        self.events.append(event)
        return event

    def finish_track(self, track_id: int) -> None:
        """Quên track đã kết thúc để không giữ bộ nhớ thừa."""
        self._sides.pop(track_id, None)

    def counts(self) -> dict[str, int]:
        """Tổng số lượt IN và OUT đã đếm."""
        return {
            "IN": sum(event.direction == "IN" for event in self.events),
            "OUT": sum(event.direction == "OUT" for event in self.events),
        }
