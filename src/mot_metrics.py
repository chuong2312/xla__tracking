"""Chỉ số chuẩn cho bài toán bám nhiều người: MOTA, MOTP (CLEAR MOT) và IDF1 (XLA 3.4).

Đây chỉ là các phép đếm trên kết quả và đáp án, không phải học máy. Giống
src/metrics.py: chỉ chấm frame có đáp án, chỉ dùng track ở trạng thái confirmed,
và một cặp (đáp án, track) khớp khi IoU ≥ ngưỡng `metrics.iou_threshold`.

- MOTA = 1 − (FN + FP + IDSW) / GT
  FN: người có trong đáp án mà không track nào khớp (bỏ sót).
  FP: track không khớp người nào (báo nhầm).
  IDSW: một người trong đáp án bị đổi sang track khác so với lần khớp trước (đổi tên nhầm).
  GT: tổng số box trong đáp án.
- MOTP = trung bình IoU của các cặp khớp: khớp càng sát thì càng gần 1.
- IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN): mỗi người trong đáp án được ghép với đúng
  MỘT track cho cả video (ghép một-một toàn cục), rồi đếm số frame mang đúng tên.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from src.assignment import assign
from src.metrics import _load_boxes, default_iou_threshold, iou


def _match_frame(truths: list[dict], preds: list[dict], last_match: dict[int, int], threshold: float) -> list[tuple[int, int]]:
    """Ghép đáp án với track trong một frame theo quy tắc CLEAR MOT.

    Bước 1: giữ lại cặp của lần khớp trước nếu IoU vẫn ≥ ngưỡng, để không tính đổi
    tên chỉ vì Hungarian chọn khác đi một chút. Bước 2: Hungarian cho phần còn lại.
    """
    pred_index = {pred["id"]: j for j, pred in enumerate(preds)}
    pairs: list[tuple[int, int]] = []
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    for i, truth in enumerate(truths):
        j = pred_index.get(last_match.get(truth["id"]))
        if j is not None and j not in used_pred and iou(truth["box"], preds[j]["box"]) >= threshold:
            pairs.append((i, j))
            used_truth.add(i)
            used_pred.add(j)
    rest_truth = [i for i in range(len(truths)) if i not in used_truth]
    rest_pred = [j for j in range(len(preds)) if j not in used_pred]
    cost = np.full((len(rest_truth), len(rest_pred)), np.inf)
    for a, i in enumerate(rest_truth):
        for b, j in enumerate(rest_pred):
            overlap = iou(truths[i]["box"], preds[j]["box"])
            if overlap >= threshold:
                cost[a, b] = 1.0 - overlap
    pairs.extend((rest_truth[a], rest_pred[b]) for a, b in assign(cost, max_cost=1.0))
    return pairs


def clear_mot(tracks_csv: str | Path, gt_csv: str | Path, iou_thr: float | None = None) -> dict[str, float | int]:
    """Tính MOTA, MOTP và các số đếm FN, FP, IDSW, GT."""
    threshold = default_iou_threshold() if iou_thr is None else float(iou_thr)
    truth_by_frame = _load_boxes(gt_csv, "gt")
    tracks_by_frame = _load_boxes(tracks_csv, "track")
    last_match: dict[int, int] = {}
    totals = Counter()
    iou_sum = 0.0
    for frame in sorted(truth_by_frame):
        truths = truth_by_frame[frame]
        preds = tracks_by_frame.get(frame, [])
        pairs = _match_frame(truths, preds, last_match, threshold)
        for i, j in pairs:
            gt_id, track_id = truths[i]["id"], preds[j]["id"]
            if gt_id in last_match and last_match[gt_id] != track_id:
                totals["idsw"] += 1
            last_match[gt_id] = track_id
            iou_sum += iou(truths[i]["box"], preds[j]["box"])
        totals["gt"] += len(truths)
        totals["matches"] += len(pairs)
        totals["fn"] += len(truths) - len(pairs)
        totals["fp"] += len(preds) - len(pairs)
    errors = totals["fn"] + totals["fp"] + totals["idsw"]
    mota = 1.0 - errors / totals["gt"] if totals["gt"] else 0.0
    motp = iou_sum / totals["matches"] if totals["matches"] else 0.0
    return {"mota": mota, "motp": motp, "fn": totals["fn"], "fp": totals["fp"], "idsw": totals["idsw"], "gt": totals["gt"]}


def idf1(tracks_csv: str | Path, gt_csv: str | Path, iou_thr: float | None = None) -> dict[str, float | int]:
    """Tính IDF1: ghép một-một mỗi người trong đáp án với một track cho cả video."""
    threshold = default_iou_threshold() if iou_thr is None else float(iou_thr)
    truth_by_frame = _load_boxes(gt_csv, "gt")
    tracks_by_frame = _load_boxes(tracks_csv, "track")
    together: Counter[tuple[int, int]] = Counter()
    gt_total = 0
    pred_total = 0
    for frame, truths in truth_by_frame.items():
        preds = tracks_by_frame.get(frame, [])
        gt_total += len(truths)
        pred_total += len(preds)
        for truth in truths:
            for pred in preds:
                if iou(truth["box"], pred["box"]) >= threshold:
                    together[(truth["id"], pred["id"])] += 1
    gt_ids = sorted({truth["id"] for truths in truth_by_frame.values() for truth in truths})
    track_ids = sorted({track_id for _, track_id in together})
    cost = np.full((len(gt_ids), len(track_ids)), np.inf)
    for (gt_id, track_id), count in together.items():
        cost[gt_ids.index(gt_id), track_ids.index(track_id)] = -count
    pairs = assign(cost, max_cost=0.0)
    idtp = sum(together[(gt_ids[i], track_ids[j])] for i, j in pairs)
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    denominator = 2 * idtp + idfp + idfn
    return {"idf1": 2 * idtp / denominator if denominator else 0.0, "idtp": idtp, "idfp": idfp, "idfn": idfn}


def mot_metrics(tracks_csv: str | Path, gt_csv: str | Path, iou_thr: float | None = None) -> dict[str, float | int]:
    """Gộp kết quả CLEAR MOT và IDF1 vào một bảng."""
    return {**clear_mot(tracks_csv, gt_csv, iou_thr), **idf1(tracks_csv, gt_csv, iou_thr)}
