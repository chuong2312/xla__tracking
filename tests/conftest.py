"""Fixture dữ liệu thật dùng chung cho test.

QĐ-31, QĐ-32: test chỉ dùng video thật, và chỉ dùng 4 video nhóm đưa (EPFL phòng thí nghiệm 4 người).
Máy nào thiếu dữ liệu thì các test cần video bị bỏ qua (skip) kèm lời nhắc; các test chỉ có con số vẫn chạy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from du_lieu_that import LAB4P, LAB4P_CALIBRATION, need


@pytest.fixture(scope="session")
def lab4p() -> Path:
    """Cảnh camera 0 + camera 1 của bộ EPFL phòng thí nghiệm 4 người (chưa có đáp án)."""
    return need(LAB4P)


@pytest.fixture(scope="session")
def calibration_4p() -> Path:
    """File hiệu chỉnh camera của bộ 4 người (EPFL đo sẵn)."""
    return need(LAB4P_CALIBRATION)
