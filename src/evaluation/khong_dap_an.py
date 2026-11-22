"""Chấm hệ thống khi video CHƯA có đáp án (QĐ-34).

Chưa có đáp án thì không tính được MOTA hay IDF1. Chỉ giảm số tên (ID) thì dễ bị "lừa": gộp nhầm nhiều
người làm một cũng giảm số tên. Vì vậy chấm cùng lúc nhiều chỉ số kéo nhau, tất cả đều đếm trên kết quả,
không phải học máy:
- so_id: số tên ở trạng thái confirmed / merged / static (ít là tốt);
- track_moi_frame: số track đang chạy trung bình mỗi frame;
- phu_nguoi: trong các khung tìm được trông như một người nguyên vẹn, tỉ lệ khung có track bám
  (IoU ≥ 0,3) (cao là tốt). Có hiệu chỉnh camera thì "nguyên vẹn" nghĩa là cao_khung / cao_mong_đợi ∈ [0,6; 1,4];
- track_ma: tỉ lệ dòng track confirmed không chồng (IoU ≥ 0,1) lên khung tìm được nào (thấp là tốt);
- nhay: số lần một track dời tâm hơn 0,5 lần chiều cao khung giữa hai frame liền nhau. Người thật
  không di chuyển nhanh như vậy, nên đây là dấu hiệu tráo người hoặc ghép nhầm (thấp là tốt);
- cao_hop_ly: trung vị cao_khung_track / cao_mong_đợi (gần 1 là tốt; cần hiệu chỉnh camera).
Chiều cao mong đợi tính từ hiệu chỉnh EPFL: đỉnh đầu = G_đầu⁻¹ · G_sàn · chân, hoặc đường ngang chứa
mọi đỉnh đầu khi camera không có G_đầu.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.common.homography import project
from src.common.utils import iou

ACTIVE = ("confirmed", "merged", "static")


def expected_height(camera: Mapping, box: Sequence[float], image_height: int) -> float | None:
    """Chiều cao người mong đợi (pixel) khi chân ở đáy giữa box; None nếu không tính được."""
    x, y, w, h = box
    floor = project(camera["ground"], [(x + w / 2.0, y + h)])
    if camera["head"] is not None:
        head_y = project(np.linalg.inv(camera["head"]), floor)[0][1]
    elif camera["head_line"] is not None:
        head_y = camera["head_line"] / 100.0 * image_height
    else:
        return None
    height = y + h - head_y
    return height if height > 10 else None


def read_rows(path: Path) -> list[dict]:
    """Đọc một file CSV kết quả thành danh sách dict."""
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _box(row: Mapping) -> tuple[int, int, int, int]:
    return tuple(int(row[k]) for k in "xywh")


def whole_person(box, camera: Mapping | None, image_height: int) -> bool:
    """Khung trông như một người nguyên vẹn (cao 0,6–1,4 lần chiều cao mong đợi); luôn đúng nếu không có hiệu chỉnh."""
    if camera is None:
        return True
    expected = expected_height(camera, box, image_height)
    return expected is not None and 0.6 <= box[3] / expected <= 1.4


def coverage(detections: list[dict], tracks: list[dict], camera, image_height: int) -> float:
    """phu_nguoi: tỉ lệ khung "người nguyên vẹn" có track đang chạy bám (IoU ≥ 0,3)."""
    by_frame = defaultdict(list)
    for row in tracks:
        if row["state"] in ACTIVE:
            by_frame[row["frame"]].append(_box(row))
    people = [(row["frame"], _box(row)) for row in detections if whole_person(_box(row), camera, image_height)]
    covered = sum(any(iou(box, t) >= 0.3 for t in by_frame[frame]) for frame, box in people)
    return covered / len(people) if people else 0.0


def ghost_share(detections: list[dict], tracks: list[dict]) -> float:
    """track_ma: tỉ lệ dòng track confirmed không chồng lên khung tìm được nào (IoU ≥ 0,1)."""
    by_frame = defaultdict(list)
    for row in detections:
        by_frame[row["frame"]].append(_box(row))
    confirmed = [row for row in tracks if row["state"] == "confirmed"]
    ghosts = sum(not any(iou(_box(row), d) >= 0.1 for d in by_frame[row["frame"]]) for row in confirmed)
    return ghosts / len(confirmed) if confirmed else 0.0


def jumps(tracks: list[dict], ratio: float = 0.5) -> int:
    """nhay: số lần tâm track dời quá `ratio` × chiều cao khung giữa hai frame liền nhau."""
    last: dict[str, tuple[int, float, float, float]] = {}
    count = 0
    for row in sorted((r for r in tracks if r["state"] in ACTIVE), key=lambda r: (r["track_id"], int(r["frame"]))):
        x, y, w, h = _box(row)
        frame, cx, cy = int(row["frame"]), x + w / 2.0, y + h / 2.0
        previous = last.get(row["track_id"])
        if previous is not None and frame == previous[0] + 1 and np.hypot(cx - previous[1], cy - previous[2]) > ratio * max(h, previous[3]):
            count += 1
        last[row["track_id"]] = (frame, cx, cy, h)
    return count


def height_plausibility(tracks: list[dict], camera, image_height: int) -> float | None:
    """cao_hop_ly: trung vị cao_khung_track / cao_mong_đợi của các dòng confirmed."""
    if camera is None:
        return None
    ratios = []
    for row in tracks:
        if row["state"] == "confirmed":
            expected = expected_height(camera, _box(row), image_height)
            if expected is not None:
                ratios.append(_box(row)[3] / expected)
    return float(np.median(ratios)) if ratios else None


def scorecard(camera_dir: Path, camera: Mapping | None, image_height: int) -> dict:
    """Mọi chỉ số của một camera, đọc từ detections.csv và tracks_linked.csv của run.py."""
    detections = read_rows(camera_dir / "detections.csv")
    tracks = read_rows(camera_dir / "tracks_linked.csv")
    frames = len({row["frame"] for row in tracks}) or 1
    active = [row for row in tracks if row["state"] in ACTIVE]
    plausible = height_plausibility(tracks, camera, image_height)
    return {"so_id": len({row["track_id"] for row in active}), "track_moi_frame": round(len(active) / frames, 2),
            "phu_nguoi": round(coverage(detections, tracks, camera, image_height), 3),
            "track_ma": round(ghost_share(detections, tracks), 3), "nhay": jumps(tracks),
            "cao_hop_ly": None if plausible is None else round(plausible, 2)}
