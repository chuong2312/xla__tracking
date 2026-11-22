"""Nối các đoạn track bị đứt của cùng một người trong MỘT camera (XLA 2.10).

Khi tracker mất dấu một người (bị che lâu, hoặc tách khỏi nhóm mà không ghép lại được),
người đó xuất hiện lại với một ID mới. Bước này chạy SAU khi đã bám người xong cả video:
1. Ứng viên: đoạn i kết thúc ở frame e_i, đoạn j bắt đầu ở frame s_j, với 0 < s_j − e_i ≤ max_gap_frames
   (hai đoạn chồng thời gian thì không thể là một người).
2. Vị trí: đoán chỗ của i ở frame s_j bằng vận tốc cuối của i (đo trên vài frame cuối);
   khoảng cách tới chỗ bắt đầu của j, chia cho đường chéo box, phải ≤ max_dist_ratio.
3. Ngoại hình: độ giống màu (src/reid/color_features.py) phải ≥ min_color_similarity.
4. Chi phí = w_m · khoảng cách chuẩn hóa + w_c · (1 − độ giống màu). Ghép một-một bằng Hungarian:
   mỗi đoạn nối với tối đa một đoạn sau. Các cặp nối thành chuỗi i → j → k, cả chuỗi lấy ID của đoạn đầu.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.common.assignment import assign
from src.reid.color_features import color_similarity

ACTIVE = ("confirmed", "merged", "static")


@dataclass
class Segment:
    """Một đoạn track: các box theo frame (chỉ lúc track đang hoạt động)."""

    track_id: int
    boxes: dict[int, tuple[int, int, int, int]]

    @property
    def first(self) -> int:
        return min(self.boxes)

    @property
    def last(self) -> int:
        return max(self.boxes)


def segments_from_rows(rows: Sequence[tuple]) -> dict[int, Segment]:
    """Gom các dòng (frame, track_id, state, x, y, w, h, n_points) thành các đoạn."""
    boxes: dict[int, dict[int, tuple]] = defaultdict(dict)
    for frame, track_id, state, x, y, w, h, *_ in rows:
        if state in ACTIVE:
            boxes[track_id][frame] = (x, y, w, h)
    return {tid: Segment(tid, b) for tid, b in boxes.items()}


def _center(box) -> np.ndarray:
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0])


def _end_velocity(segment: Segment, frames: int) -> np.ndarray:
    """Vận tốc (pixel/frame) đo trên `frames` frame cuối của đoạn."""
    times = sorted(segment.boxes)[-(frames + 1):]
    if len(times) < 2:
        return np.zeros(2)
    return (_center(segment.boxes[times[-1]]) - _center(segment.boxes[times[0]])) / (times[-1] - times[0])


def link_cost(before: Segment, after: Segment, colors: Mapping[int, dict | None], cfg: Mapping) -> float:
    """Chi phí nối đoạn `before` với đoạn `after`; np.inf nếu không được phép."""
    gap = after.first - before.last
    if gap <= 0 or gap > int(cfg["max_gap_frames"]):
        return np.inf
    last_box = before.boxes[before.last]
    predicted = _center(last_box) + gap * _end_velocity(before, int(cfg["velocity_frames"]))
    diagonal = max(float(np.hypot(last_box[2], last_box[3])), 1.0)
    distance = float(np.linalg.norm(predicted - _center(after.boxes[after.first]))) / diagonal
    if distance > float(cfg["max_dist_ratio"]):
        return np.inf
    similarity = color_similarity(colors.get(before.track_id), colors.get(after.track_id))
    if similarity < float(cfg["min_color_similarity"]):
        return np.inf
    return float(cfg["weight_motion"]) * distance + float(cfg["weight_color"]) * (1.0 - similarity)


def link_tracklets(segments: Mapping[int, Segment], colors: Mapping[int, dict | None], cfg: Mapping) -> dict[int, int]:
    """Trả về {ID cũ: ID mới}; ID mới là ID của đoạn đầu tiên trong chuỗi."""
    ids = sorted(segments)
    cost = np.array([[link_cost(segments[a], segments[b], colors, cfg) for b in ids] for a in ids]).reshape(len(ids), len(ids))
    nxt = {ids[i]: ids[j] for i, j in assign(cost, max_cost=float(cfg["max_cost"]))}
    previous = {after: before for before, after in nxt.items()}
    mapping = {}
    for track_id in ids:
        root = track_id
        while root in previous:
            root = previous[root]
        mapping[track_id] = root
    return mapping


def relabel(rows: Sequence[tuple], mapping: Mapping[int, int]) -> list[tuple]:
    """Đổi track_id trong các dòng theo `mapping` (ID không có trong mapping giữ nguyên)."""
    return [(row[0], mapping.get(row[1], row[1]), *row[2:]) for row in rows]
