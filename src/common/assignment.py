"""Ghép cặp một-một tối ưu bằng thuật toán Hungarian (bài toán phân công).

Dùng chung cho bám người (ghép track với detection) và Re-ID (ghép người ở cam A
với người ở cam B). Đầu vào là ma trận chi phí cost[i][j]: số càng nhỏ thì cặp
(i, j) càng hợp. Hungarian chọn các cặp một-một sao cho TỔNG chi phí nhỏ nhất,
khác với ghép tham lam (chọn cặp tốt nhất trước), vốn có thể làm các cặp sau bị sai.

Hai điểm cần nhớ khi dùng `scipy.optimize.linear_sum_assignment` (XLA 2.9):
1. Hàm luôn ghép đủ số cặp có thể, kể cả cặp rất tệ, nên phải lọc lại theo ngưỡng
   SAU khi gán.
2. Nếu một hàng chứa toàn vô cùng thì hàm báo lỗi "infeasible". Vì vậy cặp bị cấm
   được đổi thành một số hữu hạn rất lớn (FORBIDDEN_COST), rồi bị loại khi lọc.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.utils import iou_matrix

FORBIDDEN_COST = 1e6


def assign(cost, max_cost: float) -> list[tuple[int, int]]:
    """Ghép hàng với cột sao cho tổng chi phí nhỏ nhất; bỏ các cặp có chi phí > max_cost.

    `cost` có thể chứa np.inf hoặc NaN để đánh dấu cặp bị cấm ghép.
    Trả về danh sách (hàng, cột), sắp theo hàng.
    """
    if max_cost >= FORBIDDEN_COST:
        raise ValueError("max_cost phải nhỏ hơn FORBIDDEN_COST, nếu không cặp bị cấm sẽ lọt qua")
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Ma trận chi phí phải có 2 chiều")
    if matrix.size == 0:
        return []
    safe = np.where(np.isfinite(matrix), matrix, FORBIDDEN_COST)
    rows, cols = linear_sum_assignment(safe)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if safe[i, j] <= max_cost]


def iou_cost(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]], min_iou: float) -> np.ndarray:
    """Ma trận chi phí 1 − IoU giữa hai danh sách box; cặp có IoU < min_iou bị cấm (np.inf).

    IoU (độ chồng khớp) = diện tích phần chung / diện tích phần hợp của hai box.
    """
    overlaps = iou_matrix(boxes_a, boxes_b)
    cost = 1.0 - overlaps
    cost[overlaps < min_iou] = np.inf
    return cost
