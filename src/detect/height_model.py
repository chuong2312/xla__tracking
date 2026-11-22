"""Mô hình chiều cao người theo vị trí chân trong ảnh (QĐ-26).

Vấn đề: người mặc áo trùng màu nền (áo đen trước tường tối) bị "thủng" giữa thân, bước tìm
người trả về hai mảnh: đầu và chân. Hai mảnh nằm trên – dưới nhau, giống hệt trường hợp hai
người đứng trước – sau trong hành lang. Vị trí "trên – dưới" không phân biệt được hai trường hợp.

Cái phân biệt được là CHIỀU CAO: camera đặt cao nhìn xuống, người đứng càng gần camera thì
chân càng thấp trong ảnh và người càng cao. Gần đúng:
    chiều cao mong đợi h(y_chân) = a · y_chân + b          (pixel)
- Hai mảnh của một người gộp lại vẫn cao khoảng h(y_chân).
- Hai người trước – sau gộp lại thì cao hơn h(y_chân) rõ rệt.

Ước lượng a, b từ chính video (thống kê, không phải học máy):
1. Chạy bước tìm người (không gộp) trên các frame đầu, lấy mọi box.
2. Chia y_chân thành các khoảng `bin_px`. Trong mỗi khoảng lấy phân vị `quantile` của chiều cao:
   mảnh người thì thấp, người nguyên vẹn thì cao, nên phân vị cao ≈ chiều cao người nguyên vẹn.
3. Kẻ đường thẳng qua các điểm đó bằng Theil–Sen, một cách kẻ đường bền với điểm lạ:
       a = trung vị độ dốc (h_j − h_i) / (y_j − y_i) của mọi cặp điểm i < j
       b = trung vị của (h_i − a · y_i)
   Khoảng ở sát mép ảnh có người bị cắt mất chân, cho điểm thấp bất thường; bình phương tối
   thiểu bị một điểm như vậy kéo lệch cả đường, còn trung vị thì không.
4. Kiểm tra đường có khớp không: độ lệch tương đối r = trung vị(|h_i − (a·y_i + b)| / h_i).
   r > `max_residual` nghĩa là video không hợp giả định "sàn phẳng, camera nhìn xuống" (ví dụ quay
   giếng trời nhiều tầng: người ở tầng trên vừa cao hơn trong ảnh vừa nhỏ hơn). Khi đó không dùng
   mô hình (trả về None), để không chặn nhầm người thật. Đo được: video hợp giả định 1–11%, S1 22%.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HeightModel:
    """h(y_chân) = a · y_chân + b; `points` là các điểm (y giữa khoảng, chiều cao phân vị) đã dùng."""

    a: float
    b: float
    points: list[tuple[float, float]]
    residual: float = 0.0

    def expected(self, foot_y: float) -> float:
        """Chiều cao người mong đợi (pixel) khi chân ở hàng `foot_y`; không nhỏ hơn 1."""
        return max(1.0, self.a * foot_y + self.b)


def envelope_points(boxes: Sequence[Sequence[int]], cfg: Mapping) -> list[tuple[float, float]]:
    """Mỗi khoảng y_chân có ≥ min_count box: (y giữa khoảng, phân vị `quantile` của chiều cao)."""
    if not boxes:
        return []
    data = np.asarray([(y + h, h) for _, y, _, h in boxes], dtype=np.float32)
    step = float(cfg["bin_px"])
    points = []
    for start in np.arange(data[:, 0].min(), data[:, 0].max() + 1, step):
        heights = data[(data[:, 0] >= start) & (data[:, 0] < start + step), 1]
        if len(heights) >= int(cfg["min_count"]):
            points.append((float(start + step / 2), float(np.percentile(heights, float(cfg["quantile"])))))
    return points


def theil_sen(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Đường thẳng h = a·y + b bền với điểm lạ: a, b đều lấy trung vị (xem đầu file)."""
    slopes = [(h2 - h1) / (y2 - y1) for i, (y1, h1) in enumerate(points) for y2, h2 in points[i + 1:] if y2 != y1]
    a = float(np.median(slopes))
    b = float(np.median([h - a * y for y, h in points]))
    return a, b


def relative_residual(points: Sequence[tuple[float, float]], a: float, b: float) -> float:
    """Trung vị của |h − (a·y + b)| / h trên các điểm: đường thẳng khớp các điểm tới đâu."""
    return float(np.median([abs(h - (a * y + b)) / max(h, 1.0) for y, h in points]))


def fit_height_model(boxes: Sequence[Sequence[int]], cfg: Mapping) -> HeightModel | None:
    """Kẻ đường Theil–Sen qua các điểm phân vị.

    None (không dùng mô hình) nếu có ít hơn 3 điểm, độ dốc âm (vô lý), hoặc đường không khớp các
    điểm (độ lệch tương đối > max_residual).
    """
    points = envelope_points(boxes, cfg)
    if len(points) < 3:
        return None
    a, b = theil_sen(points)
    residual = relative_residual(points, a, b)
    if a <= 0 or residual > float(cfg["max_residual"]):
        return None
    return HeightModel(float(a), float(b), points, residual)


def tall_enough(box: Sequence[float], model: HeightModel, min_ratio: float, image_shape: tuple[int, int], border_px: float) -> bool:
    """Box có đủ cao để là cả một người không (QĐ-27).

    Đúng khi  h ≥ min_ratio · h_mong_đợi(y + h),  hoặc khi box chạm mép ảnh (cách mép ≤ border_px):
    người đang bước vào khung hình bị cắt một phần nên thấp, nhưng vẫn là người thật.
    Mảnh đầu hoặc mảnh chân (áo trùng màu nền) thường chỉ cao 20–50% chiều cao mong đợi.
    """
    x, y, w, h = box
    height, width = image_shape
    at_border = x <= border_px or y <= border_px or x + w >= width - border_px or y + h >= height - border_px
    return at_border or h >= min_ratio * model.expected(y + h)


def save(model: HeightModel, path: Path) -> None:
    """Ghi mô hình ra JSON để xem lại và giải thích."""
    data = {"a": model.a, "b": model.b, "do_lech_tuong_doi": model.residual, "points": model.points}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> HeightModel:
    """Đọc mô hình từ JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return HeightModel(float(data["a"]), float(data["b"]), [tuple(p) for p in data["points"]], float(data.get("do_lech_tuong_doi", 0.0)))
