"""Vẽ khung hình cho video tổng hợp v2 và tính đáp án phần nhìn thấy của từng người.

Cách tính che khuất: vẽ người ở xa trước, người ở gần sau, nên người gần đè lên người
xa giống như ngoài đời. Song song với ảnh màu, ta giữ một "bản đồ ai ở pixel nào"
(label map): mỗi pixel ghi mã người đang hiện ở đó. Phần nhìn thấy của một người là
các pixel còn mang mã của họ sau khi vẽ xong.
"""
from __future__ import annotations

import numpy as np
import cv2

from src.common.homography import project
from tools.synthetic_scene import IMAGE_SIZE, Layout, Person, rect_corners

NOISE_STD = 2.0
PATTERN_AMPLITUDE = 45
PATTERN_PERIOD_PX = 6
HEAD_RATIO = 0.15
TOP_RATIO = 0.45
SKIN_BGR = (140, 170, 210)
VISIBLE_FULLY = 0.9
VISIBLE_PARTLY = 0.5
NO_PERSON = -1


def _floor_polygon(H: np.ndarray, corners_m) -> np.ndarray:
    """Chiếu một đa giác trên sàn (mét) sang ảnh, trả về mảng điểm nguyên cho cv2."""
    return np.round(project(H, corners_m)).astype(np.int32).reshape(-1, 1, 2)


def make_background(layout: Layout, H: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ảnh nền cố định của một camera: tường có đồ vật, sàn kẻ ô, băng keo.

    Đồ vật và đường kẻ tạo góc cạnh trên nền, giống cảnh thật, để thử xem thuật toán
    có lấy nhầm điểm nền hay không.
    """
    width, height = IMAGE_SIZE
    gradient = np.linspace(150, 110, height, dtype=np.float32)[:, None, None]
    image = np.repeat(np.repeat(gradient, width, axis=1), 3, axis=2).astype(np.uint8)
    for _ in range(8):
        x, y = int(rng.integers(0, width - 60)), int(rng.integers(0, height // 2))
        w, h = int(rng.integers(30, 90)), int(rng.integers(20, 70))
        color = tuple(int(c) for c in rng.integers(60, 200, 3))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (40, 40, 40), 2)
    x_min, y_min, x_max, y_max = layout.floor_rect_m
    cv2.fillPoly(image, [_floor_polygon(H, rect_corners(layout.floor_rect_m))], (150, 175, 190))
    for x in np.arange(x_min, x_max + 0.01, 1.0):
        cv2.polylines(image, [_floor_polygon(H, [(x, y_min), (x, y_max)])], False, (120, 140, 150), 1)
    for y in np.arange(y_min, y_max + 0.01, 1.0):
        cv2.polylines(image, [_floor_polygon(H, [(x_min, y), (x_max, y)])], False, (120, 140, 150), 1)
    for mx, my in layout.tape_marks_m:
        square = [(mx - 0.08, my - 0.08), (mx + 0.08, my - 0.08), (mx + 0.08, my + 0.08), (mx - 0.08, my + 0.08)]
        cv2.fillPoly(image, [_floor_polygon(H, square)], (0, 140, 255))
    return image


def person_patch(person: Person, width: int, height: int) -> np.ndarray:
    """Ảnh của một người cỡ width × height: đầu màu da, áo và quần có hoa văn riêng.

    Hoa văn tạo góc cạnh trên người (để Lucas-Kanade có điểm bám) và giúp phân biệt
    ngoại hình. Hai người cùng `pattern` và cùng màu thì trông giống hệt nhau.
    """
    patch = np.zeros((height, width, 3), dtype=np.int16)
    head_end = int(height * HEAD_RATIO)
    top_end = head_end + int(height * TOP_RATIO)
    patch[:head_end] = SKIN_BGR
    patch[head_end:top_end] = person.top_bgr
    patch[top_end:] = person.bottom_bgr
    rows, cols = np.indices((height, width))
    period = PATTERN_PERIOD_PX
    masks = {
        "soc_ngang": (rows // period) % 2 == 0,
        "soc_doc": (cols // period) % 2 == 0,
        "o_vuong": ((rows // period) + (cols // period)) % 2 == 0,
        "cham": ((rows % period) < 2) & ((cols % period) < 2),
    }
    texture = masks[person.pattern].astype(np.int16) * PATTERN_AMPLITUDE
    patch[head_end:] += texture[head_end:, :, None]
    return np.clip(patch, 0, 255).astype(np.uint8)


def render_frame(background: np.ndarray, people_boxes, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Vẽ một khung hình. `people_boxes` là danh sách (Person, box đầy đủ dạng float).

    Trả về (ảnh BGR, bản đồ ai ở pixel nào). Người có chân thấp hơn trong ảnh (gần camera
    hơn) được vẽ sau nên đè lên người ở xa.
    """
    height, width = background.shape[:2]
    frame = background.copy()
    labels = np.full((height, width), NO_PERSON, dtype=np.int16)
    for person, box in sorted(people_boxes, key=lambda item: item[1][1] + item[1][3]):
        x, y, w, h = (int(round(v)) for v in box)
        if w <= 0 or h <= 0:
            continue
        patch = person_patch(person, w, h)
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + w), min(height, y + h)
        if right <= left or bottom <= top:
            continue
        frame[top:bottom, left:right] = patch[top - y:bottom - y, left - x:right - x]
        labels[top:bottom, left:right] = person.gt_id
    noise = rng.normal(0.0, NOISE_STD, frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, labels


def occlusion_level(visible_ratio: float) -> int:
    """Mức che theo tỉ lệ nhìn thấy: ≥ 90% là 0, từ 50% đến dưới 90% là 1, dưới 50% là 2."""
    if visible_ratio >= VISIBLE_FULLY:
        return 0
    if visible_ratio >= VISIBLE_PARTLY:
        return 1
    return 2


def visible_annotations(labels: np.ndarray, people_boxes) -> list[tuple[int, int, int, int, int, int]]:
    """Đáp án của một frame: (gt_id, x, y, w, h, occluded), box chỉ bao phần nhìn thấy (XLA 3.3).

    Tỉ lệ nhìn thấy = số pixel còn mang mã người đó / diện tích box nằm trong ảnh.
    Phần ra ngoài mép ảnh không tính là bị che. Người không còn pixel nào thì không ghi.
    """
    height, width = labels.shape
    rows = []
    for person, box in people_boxes:
        x, y, w, h = (int(round(v)) for v in box)
        inside_w = max(0, min(width, x + w) - max(0, x))
        inside_h = max(0, min(height, y + h) - max(0, y))
        ys, xs = np.nonzero(labels == person.gt_id)
        if inside_w * inside_h == 0 or len(xs) == 0:
            continue
        level = occlusion_level(len(xs) / (inside_w * inside_h))
        left, top = int(xs.min()), int(ys.min())
        rows.append((person.gt_id, left, top, int(xs.max()) - left + 1, int(ys.max()) - top + 1, level))
    return rows
