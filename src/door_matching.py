"""Ghép sự kiện qua cửa giữa hai camera theo thời gian (QĐ-18).

Dùng cho bố trí "hành lang + cửa" (QĐ-17): hai camera cùng thấy người đi qua cửa.
Khi hai video đã đồng bộ giờ (vỗ tay lúc bắt đầu quay), một lượt VÀO ở camera A và
một lượt VÀO ở camera B xảy ra gần như cùng lúc thì nhiều khả năng là cùng một người.
Cách này không cần nhìn quần áo, nên phân biệt được cả người mặc giống nhau. Nó chỉ
bế tắc khi hai người qua cửa gần như cùng lúc; khi đó phải nhờ ngoại hình và vị trí sàn.

Chi phí ghép giữa sự kiện a (camera A) và b (camera B) là độ lệch thời gian
    gap = |frame_a − frame_b| / fps   (giây)
Cặp bị cấm khi khác chiều (một VÀO, một RA) hoặc gap > max_gap_s.
Ghép một-một bằng Hungarian (src/assignment.py) để TỔNG độ lệch nhỏ nhất.
Số frame của hai camera phải cùng một mốc thời gian, tức đã cộng `frame_offset` của camera B.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.assignment import assign
from src.door import DoorEvent


@dataclass(frozen=True)
class EventMatch:
    """Một cặp sự kiện được ghép: cùng chiều, xảy ra gần như cùng lúc ở hai camera."""

    event_a: DoorEvent
    event_b: DoorEvent
    gap_s: float


def match_door_events(
    events_a: Sequence[DoorEvent],
    events_b: Sequence[DoorEvent],
    fps: float,
    max_gap_s: float,
) -> list[EventMatch]:
    """Ghép sự kiện qua cửa của camera A với camera B; trả về các cặp, sắp theo thời gian."""
    if fps <= 0:
        raise ValueError("fps phải dương")
    if max_gap_s < 0:
        raise ValueError("reid.door_event_max_gap_s phải không âm")
    cost = np.full((len(events_a), len(events_b)), np.inf)
    for i, event_a in enumerate(events_a):
        for j, event_b in enumerate(events_b):
            gap = abs(event_a.frame - event_b.frame) / fps
            if event_a.direction == event_b.direction and gap <= max_gap_s:
                cost[i, j] = gap
    pairs = assign(cost, max_cost=max_gap_s)
    matches = [EventMatch(events_a[i], events_b[j], float(cost[i, j])) for i, j in pairs]
    return sorted(matches, key=lambda match: match.event_a.frame)


def track_links(matches: Sequence[EventMatch]) -> dict[int, int]:
    """Từ các cặp sự kiện, suy ra track nào ở camera A ứng với track nào ở camera B."""
    return {match.event_a.track_id: match.event_b.track_id for match in matches}
