"""Phân loại detection TRƯỚC khi ghép với track (XLA 2.3).

- Blob tách: một người bị vỡ thành nhiều detection (ví dụ thân và chân rời nhau). Nhận ra
  khi có từ 2 detection nằm phần lớn trong box dự đoán của CÙNG MỘT track và không thuộc
  track nào khác. Các detection đó được gộp lại thành một box bao chung.
- Blob gộp: một detection chứa từ 2 người. Nhận ra khi có từ 2 track mà tâm box dự đoán
  nằm trong detection và phần lớn box dự đoán (≥ merge_coverage) nằm trong detection.
  Detection đó bị loại khỏi bước ghép; các track liên quan chuyển sang trạng thái merged.
- Track mảnh: hai track cùng nằm trong một detection, xếp trên – dưới và gần như không chồng
  nhau theo chiều dọc (một track chỉ có đầu, một track chỉ có chân). Đó là hai mảnh của cùng
  một người từng bị tách; track trẻ hơn bị bỏ, không tính là blob gộp.
- Còn lại là detection sạch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

Box = tuple[int, int, int, int]


def coverage(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Tỉ lệ diện tích của `inner` nằm trong `outer` (0–1)."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    if iw <= 0 or ih <= 0:
        return 0.0
    overlap_w = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
    overlap_h = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
    return overlap_w * overlap_h / (iw * ih)


def center_inside(box: Sequence[float], container: Sequence[float]) -> bool:
    """Tâm của `box` có nằm trong `container` không."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    left, top, width, height = container
    return left <= cx <= left + width and top <= cy <= top + height


def union_box(boxes: Sequence[Sequence[float]]) -> Box:
    """Box nhỏ nhất bao tất cả các box."""
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return (int(left), int(top), int(right - left), int(bottom - top))


def stacked_vertically(a: Sequence[float], b: Sequence[float], min_overlap: float = 0.5) -> bool:
    """Hai box có nằm chồng theo chiều dọc không: phần chung theo chiều ngang ≥ min_overlap × bề rộng box hẹp hơn.

    Mảnh thân và mảnh chân của một người nằm trên dưới nhau; hai người đi cạnh nhau nằm ngang hàng.
    """
    overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    return overlap >= min_overlap * min(a[2], b[2])


def join_split_detections(detections: Sequence[Box], predicted: Mapping[int, Box], min_coverage: float) -> list[Box]:
    """Gộp các mảnh detection thuộc cùng một track dự đoán thành một box.

    Chỉ gộp khi các mảnh nằm chồng theo chiều dọc (thân trên – chân), không gộp hai mảnh nằm
    ngang hàng, vì đó thường là hai người đi cạnh nhau.
    """
    owner: dict[int, int] = {}
    for index, detection in enumerate(detections):
        owners = [tid for tid, box in predicted.items() if coverage(detection, box) >= min_coverage]
        if len(owners) == 1:
            owner[index] = owners[0]
    groups: dict[int, list[int]] = {}
    for index, track_id in owner.items():
        groups.setdefault(track_id, []).append(index)
    groups = {tid: members for tid, members in groups.items()
              if all(stacked_vertically(detections[i], detections[j]) for i in members for j in members if i < j)}
    joined = {i for members in groups.values() if len(members) >= 2 for i in members}
    result = [d for i, d in enumerate(detections) if i not in joined]
    result += [union_box([detections[i] for i in members]) for members in groups.values() if len(members) >= 2]
    return result


def vertical_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Phần chung theo chiều dọc, chia cho chiều cao box thấp hơn (0–1)."""
    overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, overlap) / max(min(a[3], b[3]), 1.0)


def find_fragments(detections: Sequence[Box], predicted: Mapping[int, Box], min_coverage: float,
                   max_vertical_overlap: float) -> set[int]:
    """Track nào chỉ là mảnh thừa của một track khác (cùng một người), cần bỏ.

    Hai track t1 (cũ hơn, id nhỏ hơn) và t2 là mảnh của nhau khi cả hai nằm phần lớn
    (≥ min_coverage) trong cùng một detection và:
        stacked_vertically(t1, t2)  và  vertical_overlap(t1, t2) ≤ max_vertical_overlap.
    Hai người đứng trước – sau thì box chồng nhau nhiều theo chiều dọc nên không bị coi là mảnh.
    """
    fragments: set[int] = set()
    for detection in detections:
        inside = sorted(tid for tid, box in predicted.items() if coverage(box, detection) >= min_coverage)
        for i, older in enumerate(inside):
            for younger in inside[i + 1:]:
                a, b = predicted[older], predicted[younger]
                if older not in fragments and stacked_vertically(a, b) and vertical_overlap(a, b) <= max_vertical_overlap:
                    fragments.add(younger)
    return fragments


def find_merged(detections: Sequence[Box], predicted: Mapping[int, Box], min_coverage: float,
                sticky_ids: set[int] | frozenset = frozenset(), sticky_coverage: float | None = None) -> dict[int, set[int]]:
    """Với mỗi detection là blob gộp, trả về tập track nằm trong nó: {chỉ số detection: {track id}}.

    Một track nằm trong detection khi tâm box dự đoán nằm trong detection và ≥ min_coverage
    diện tích box nằm trong detection. Track ĐANG ở trạng thái gộp (`sticky_ids`) dùng ngưỡng
    thấp hơn `sticky_coverage` và không cần tâm nằm trong (ngưỡng có độ trễ): vị trí dự đoán
    của nó trôi dần ra mép blob, nếu dùng cùng ngưỡng thì nó chập chờn "ra – vào" và bị tính là mất.
    """
    merged: dict[int, set[int]] = {}
    for index, detection in enumerate(detections):
        inside = set()
        for tid, box in predicted.items():
            if tid in sticky_ids and sticky_coverage is not None:
                if coverage(box, detection) >= sticky_coverage:
                    inside.add(tid)
            elif center_inside(box, detection) and coverage(box, detection) >= min_coverage:
                inside.add(tid)
        if len(inside) >= 2:
            merged[index] = inside
    return merged
