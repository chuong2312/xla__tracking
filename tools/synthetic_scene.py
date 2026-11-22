"""Các khối dựng "sân khấu" cho video tổng hợp v2: người, bố trí sàn – camera, kịch bản (QĐ-16, QĐ-22).

Mọi vị trí của người được tính trên MẶT SÀN, đơn vị mét. Mỗi camera có một
homography "sàn → ảnh", nên cùng một người có đáp án đúng ở mọi camera.
Dữ liệu cụ thể (có những bố trí nào, kịch bản nào) nằm ở tools/synthetic_scenarios.py.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from src.homography import project

IMAGE_SIZE = (640, 360)
EMPTY_SECONDS = 2.0
DURATION_SECONDS = 12.0
PERSON_WIDTH_M = 0.5
PERSON_HEIGHT_M = 1.7
HEIGHT_FORESHORTENING = 0.8
MAX_PERSON_HEIGHT_PX = 220
CAMERAS = ("cam_a", "cam_b")

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class Person:
    """Một người trong cảnh: ngoại hình và các mốc (giây, x mét, y mét) trên đường đi."""

    gt_id: int
    top_bgr: tuple[int, int, int]
    bottom_bgr: tuple[int, int, int]
    pattern: str
    waypoints: tuple[tuple[float, float, float], ...]

    def position(self, t: float) -> Point | None:
        """Vị trí trên sàn ở thời điểm t, nội suy tuyến tính giữa hai mốc liền kề.

        Trả về None nếu người chưa vào cảnh (t < mốc đầu) hoặc đã đi khỏi (t > mốc cuối).
        """
        if t < self.waypoints[0][0] or t > self.waypoints[-1][0]:
            return None
        for (t0, x0, y0), (t1, x1, y1) in zip(self.waypoints, self.waypoints[1:]):
            if t0 <= t <= t1:
                ratio = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return (x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0))
        _, x, y = self.waypoints[-1]
        return (x, y)


@dataclass(frozen=True)
class Layout:
    """Một cách bố trí sàn và camera.

    - floor_rect_m: (x_min, y_min, x_max, y_max), phần sàn được tô màu.
    - camera_views: với mỗi camera, 4 điểm trên sàn và 4 điểm tương ứng trong ảnh,
      đủ để tính homography sàn → ảnh.
    - tape_marks_m: các điểm băng keo (dùng để thử công cụ click điểm sàn).
    """

    name: str
    floor_rect_m: tuple[float, float, float, float]
    camera_views: Mapping[str, tuple[Quad, Quad]]
    tape_marks_m: tuple[Point, ...]


@dataclass(frozen=True)
class Scenario:
    """Một kịch bản: bố trí nào, những người nào đi lại ra sao."""

    name: str
    layout: Layout
    people: tuple[Person, ...]
    description: str


def rect_corners(rect: tuple[float, float, float, float]) -> list[Point]:
    """4 góc của hình chữ nhật trên sàn, theo vòng (x_min,y_min) → (x_max,y_min) → (x_max,y_max) → (x_min,y_max)."""
    x_min, y_min, x_max, y_max = rect
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def floor_to_image(layout: Layout, camera: str) -> np.ndarray:
    """Homography sàn (mét) → ảnh (pixel) của một camera, tính từ 4 cặp điểm."""
    floor_quad, image_quad = layout.camera_views[camera]
    return cv2.getPerspectiveTransform(np.float32(floor_quad), np.float32(image_quad)).astype(np.float64)


def pixels_per_meter(H: np.ndarray, x: float, y: float, delta: float = 0.05) -> float:
    """Tỉ lệ pixel trên mét ở chỗ (x, y) trên sàn: s = √|det J|.

    J là ma trận đạo hàm của phép chiếu sàn → ảnh, tính xấp xỉ bằng sai phân nhỏ `delta`.
    Chỗ gần camera s lớn (người trông to), chỗ xa camera s nhỏ.
    """
    base, step_x, step_y = project(H, [(x, y), (x + delta, y), (x, y + delta)])
    jacobian = np.column_stack([(step_x - base) / delta, (step_y - base) / delta])
    return math.sqrt(abs(float(np.linalg.det(jacobian))))


def person_box(H: np.ndarray, x: float, y: float) -> tuple[float, float, float, float]:
    """Box đầy đủ (chưa xét che khuất) của người đứng ở (x, y): chân ở giữa cạnh dưới."""
    foot_x, foot_y = project(H, [(x, y)])[0]
    scale = pixels_per_meter(H, x, y)
    width = PERSON_WIDTH_M * scale
    height = min(PERSON_HEIGHT_M * scale * HEIGHT_FORESHORTENING, MAX_PERSON_HEIGHT_PX)
    return (foot_x - width / 2.0, foot_y - height, width, height)


def n_frames(fps: float) -> int:
    """Số frame của mỗi kịch bản."""
    return int(round(DURATION_SECONDS * fps))
