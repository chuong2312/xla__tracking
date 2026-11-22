"""Kiểm thử phân loại blob của tracker mới. Chỉ có con số (tọa độ khung), không có hình ảnh.

Các test cần hình ảnh chạy trên video thật, xem test_that_tim_nguoi.py và test_that_bam_nguoi.py (QĐ-31).
"""
from src.tracking.blob_analysis import find_merged, join_split_detections


def test_split_pieces_are_joined_only_when_stacked_vertically():
    predicted = {1: (10, 10, 30, 80)}
    stacked = join_split_detections([(12, 10, 26, 35), (14, 50, 22, 38)], predicted, 0.7)
    side_by_side = join_split_detections([(10, 10, 14, 80), (26, 10, 14, 80)], predicted, 0.7)
    assert stacked == [(12, 10, 26, 78)]
    assert len(side_by_side) == 2


def test_blob_containing_two_tracks_is_merged():
    predicted = {1: (10, 10, 20, 60), 2: (25, 12, 20, 60)}
    assert find_merged([(8, 8, 40, 66)], predicted, 0.6) == {0: {1, 2}}


def test_track_already_merged_stays_in_blob_with_lower_coverage():
    """Ngưỡng có độ trễ: track đang gộp chỉ cần 30% box nằm trong blob là vẫn tính trong blob."""
    predicted = {1: (10, 10, 20, 60), 2: (36, 10, 20, 60)}
    blob = [(8, 8, 36, 66)]
    assert find_merged(blob, predicted, 0.6) == {}
    assert find_merged(blob, predicted, 0.6, sticky_ids={2}, sticky_coverage=0.3) == {0: {1, 2}}
