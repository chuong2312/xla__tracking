"""Chấm điểm toàn hệ thống khi có đáp án (chế độ end-to-end, XLA 2.10).

Khác chế độ oracle: track ở đây do chương trình tự tạo, nên mỗi track phải được gắn với một
người trong đáp án trước đã:
- Ở mỗi frame có đáp án, ghép track với người trong đáp án bằng Hungarian trên IoU (≥ ngưỡng).
- Mỗi track được gắn với người mà nó khớp nhiều frame nhất (bỏ phiếu đa số).
- Nếu một người bị cắt thành nhiều track, "track chính" của người đó là track khớp lâu nhất.
Rồi chấm:
- Nhận dạng lại: với mỗi người có ở cả hai camera, đúng nếu track chính ở A và track chính ở B
  có cùng global_id (khác −1).
- Đếm cửa: một lượt của hệ thống khớp với đáp án nếu cùng người, cùng chiều và lệch không quá
  `tolerance_s` giây.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.door import DoorEvent
from src.metrics import _load_boxes, _match_boxes, default_iou_threshold


def map_tracks_to_gt(tracks_csv: str | Path, gt_csv: str | Path, iou_thr: float | None = None) -> tuple[dict[int, int], dict[int, int]]:
    """Trả về ({track: người trong đáp án}, {người: track chính}) theo bỏ phiếu đa số."""
    threshold = default_iou_threshold() if iou_thr is None else iou_thr
    truth = _load_boxes(gt_csv, "gt")
    tracked = _load_boxes(tracks_csv, "track")
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame, truths in truth.items():
        preds = tracked.get(frame, [])
        for i, j in _match_boxes(truths, preds, threshold):
            votes[preds[j]["id"]][truths[i]["id"]] += 1
    track_to_gt = {track_id: counts.most_common(1)[0][0] for track_id, counts in votes.items()}
    main_track: dict[int, int] = {}
    for track_id, gt_id in track_to_gt.items():
        current = main_track.get(gt_id)
        if current is None or votes[track_id][gt_id] > votes[current][gt_id]:
            main_track[gt_id] = track_id
    return track_to_gt, main_track


def read_pairs(gt_reid_csv: str | Path) -> list[tuple[int, int]]:
    """Các cặp (người ở cam A, người ở cam B) trong đáp án Re-ID."""
    with Path(gt_reid_csv).open(encoding="utf-8") as stream:
        return [(int(r["cam_a_id"]), int(r["cam_b_id"])) for r in csv.DictReader(stream)]


def evaluate_reid(global_rows: Sequence[tuple[str, int, int]], main_a: Mapping[int, int], main_b: Mapping[int, int],
                  gt_pairs: Sequence[tuple[int, int]]) -> dict[str, int | str]:
    """Số người được ghép đúng giữa hai camera, theo track chính của mỗi người."""
    global_ids = {(cam, track): gid for cam, track, gid in global_rows}
    correct = 0
    for person_a, person_b in gt_pairs:
        a = global_ids.get(("cam_a", main_a.get(person_a)), -1)
        b = global_ids.get(("cam_b", main_b.get(person_b)), -1)
        correct += int(a != -1 and a == b)
    return {"correct": correct, "total": len(gt_pairs), "accuracy_str": f"{correct}/{len(gt_pairs)}"}


def evaluate_door(events: Sequence[DoorEvent], track_to_gt: Mapping[int, int], gt_events_csv: str | Path,
                  fps: float, tolerance_s: float) -> dict[str, int | str]:
    """So các lượt qua cửa của hệ thống với đáp án."""
    with Path(gt_events_csv).open(encoding="utf-8") as stream:
        truth = [(int(r["frame"]), int(r["gt_id"]), r["direction"]) for r in csv.DictReader(stream)]
    unused = list(truth)
    matched = 0
    for event in events:
        person = track_to_gt.get(event.track_id)
        for candidate in unused:
            frame, gt_id, direction = candidate
            if gt_id == person and direction == event.direction and abs(frame - event.frame) / fps <= tolerance_s:
                unused.remove(candidate)
                matched += 1
                break
    return {
        "dap_an_vao": sum(d == "IN" for _, _, d in truth), "dap_an_ra": sum(d == "OUT" for _, _, d in truth),
        "he_thong_vao": sum(e.direction == "IN" for e in events), "he_thong_ra": sum(e.direction == "OUT" for e in events),
        "khop_dung": f"{matched}/{len(truth)}",
    }
