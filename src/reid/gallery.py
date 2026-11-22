"""Chọn bộ ảnh mẫu (gallery) cho mỗi track: K lần xuất hiện "đẹp" nhất (bước F).

Một lần xuất hiện được chọn khi thỏa cả ba điều:
- sạch: không bị che (khi dùng đáp án: occluded = 0; khi dùng track thật: không nằm trong blob gộp);
- không chạm mép ảnh, vì người bị cắt ở mép thì ngoại hình không đầy đủ;
- cách mọi lần đã chọn ít nhất `min_gap_frames` frame, để gallery có nhiều tư thế khác nhau.
Trong các lần thỏa điều kiện, ưu tiên box có diện tích lớn (người to, rõ chi tiết).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """Một lần xuất hiện của track: frame, box (x, y, w, h) và có sạch hay không."""

    frame: int
    box: tuple[int, int, int, int]
    clean: bool


def touches_border(box: Sequence[int], image_size: Sequence[int], border_px: int) -> bool:
    """Box có chạm (hoặc gần trong vòng `border_px` pixel) mép ảnh không."""
    x, y, w, h = box
    width, height = image_size
    return x <= border_px or y <= border_px or x + w >= width - border_px or y + h >= height - border_px


def select_gallery(
    observations: Sequence[Observation],
    image_size: Sequence[int],
    size: int,
    min_gap_frames: int,
    border_px: int,
) -> list[Observation]:
    """Chọn tối đa `size` lần xuất hiện theo các tiêu chí ở đầu file, sắp lại theo thời gian."""
    candidates = [o for o in observations if o.clean and not touches_border(o.box, image_size, border_px)]
    candidates.sort(key=lambda o: o.box[2] * o.box[3], reverse=True)
    chosen: list[Observation] = []
    for observation in candidates:
        if all(abs(observation.frame - other.frame) >= min_gap_frames for other in chosen):
            chosen.append(observation)
        if len(chosen) == size:
            break
    return sorted(chosen, key=lambda o: o.frame)
