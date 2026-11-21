"""Kiểm thử ghép cặp Hungarian dùng chung (src/assignment.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.assignment import FORBIDDEN_COST, assign, iou_cost


def test_empty_matrix_gives_no_pairs():
    assert assign(np.empty((0, 3)), max_cost=1.0) == []


def test_hungarian_minimises_total_cost_where_greedy_fails():
    # Ghép tham lam sẽ lấy (0,0)=1 trước rồi buộc phải lấy (1,1)=100, tổng 101.
    # Hungarian lấy (0,1)+(1,0) = 4.
    assert assign([[1, 2], [2, 100]], max_cost=10) == [(0, 1), (1, 0)]


def test_non_square_matrix_leaves_extra_column_unmatched():
    pairs = assign([[0.1, 0.9, 0.5], [0.8, 0.2, 0.7]], max_cost=1.0)
    assert pairs == [(0, 0), (1, 1)]


def test_row_with_only_forbidden_pairs_does_not_crash():
    pairs = assign([[np.inf, np.inf], [0.3, 0.4]], max_cost=1.0)
    assert pairs == [(1, 0)]


def test_pairs_above_max_cost_are_removed_after_assignment():
    assert assign([[0.9, 0.95], [0.97, 0.99]], max_cost=0.5) == []


def test_max_cost_must_stay_below_forbidden_cost():
    with pytest.raises(ValueError):
        assign([[1.0]], max_cost=FORBIDDEN_COST)


def test_iou_cost_forbids_pairs_below_min_iou():
    cost = iou_cost([(0, 0, 10, 10)], [(0, 0, 10, 10), (50, 50, 10, 10)], min_iou=0.1)
    assert cost[0, 0] == pytest.approx(0.0)
    assert np.isinf(cost[0, 1])
