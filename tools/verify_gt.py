"""Kiểm tra file đáp án (ground truth) và xuất video có vẽ box để xem bằng mắt.

Cách dùng:
    python tools/verify_gt.py --video <video> --gt <gt.csv> --out <video_kiem_tra.mp4> [--size 640 360]

Đọc được cả định dạng cũ (`frame,gt_id,x,y,w,h`) lẫn định dạng mới có thêm
`occluded,keyframe` (QĐ-15). File cũ được coi là occluded = 0 và keyframe = 1.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config  # noqa: E402
from src.utils import iter_video  # noqa: E402

GtRow = tuple[int, int, int, int, int, int, int, int]
OCCLUSION_COLORS = {0: (0, 220, 0), 1: (0, 200, 255), 2: (0, 0, 255)}


def read_gt(path: str | Path) -> list[GtRow]:
    """Đọc file đáp án; thiếu cột occluded thì coi là 0, thiếu cột keyframe thì coi là 1."""
    rows: list[GtRow] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            occluded = int(record.get("occluded") or 0)
            keyframe = int(record.get("keyframe") or 1)
            box = tuple(int(record[key]) for key in ("x", "y", "w", "h"))
            rows.append((int(record["frame"]), int(record["gt_id"]), *box, occluded, keyframe))
    return rows


def validate_gt(rows: list[GtRow], width: int, height: int) -> list[str]:
    """Trả về danh sách cảnh báo: box ra ngoài khung, ID bị hở frame, giá trị cột lạ."""
    warnings: list[str] = []
    frames_by_id: dict[int, list[int]] = defaultdict(list)
    for frame, pid, x, y, w, h, occluded, keyframe in rows:
        frames_by_id[pid].append(frame)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            warnings.append(f"Box ngoài khung hoặc suy biến: frame={frame}, gt_id={pid}, box={(x, y, w, h)}")
        if occluded not in (0, 1, 2):
            warnings.append(f"occluded phải là 0, 1 hoặc 2: frame={frame}, gt_id={pid}, giá trị={occluded}")
        if keyframe not in (0, 1):
            warnings.append(f"keyframe phải là 0 hoặc 1: frame={frame}, gt_id={pid}, giá trị={keyframe}")
    for pid, frames in frames_by_id.items():
        unique = sorted(set(frames))
        if any(b - a > 1 for a, b in zip(unique, unique[1:])):
            warnings.append(f"gt_id={pid} có lỗ hổng frame")
    return warnings


def write_overlay(video: str, rows: list[GtRow], out: str, size: tuple[int, int]) -> dict[int, set[int]]:
    """Vẽ box đáp án lên video, màu theo mức che (xanh 0, vàng 1, đỏ 2)."""
    source = cv2.VideoCapture(video)
    fps = source.get(cv2.CAP_PROP_FPS)
    source.release()
    if fps <= 0:
        raise ValueError(f"Không đọc được FPS hợp lệ từ video: {video}")
    grouped: dict[int, list[GtRow]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được video: {out}")
    seen: dict[int, set[int]] = defaultdict(set)
    try:
        for frame_index, frame in iter_video(video, size):
            for _, pid, x, y, w, h, occluded, _ in grouped.get(frame_index, []):
                color = OCCLUSION_COLORS.get(occluded, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{pid} che{occluded}", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                seen[pid].add(frame_index)
            writer.write(frame)
    finally:
        writer.release()
    return seen


def main() -> None:
    """In số frame của từng ID và các cảnh báo; ghi video kiểm tra."""
    parser = argparse.ArgumentParser(description="Kiểm tra file đáp án")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    size = tuple(args.size) if args.size else (cfg.scene.resize_width, cfg.scene.resize_height)
    rows = read_gt(args.gt)
    seen = write_overlay(args.video, rows, args.out, size)
    print(f"Số gt_id: {len(seen)}")
    for pid, frames in sorted(seen.items()):
        print(f"gt_id={pid}: {len(frames)} frame")
    for warning in validate_gt(rows, *size):
        print(f"CẢNH BÁO: {warning}")


if __name__ == "__main__":
    main()
