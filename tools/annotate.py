"""Công cụ gán nhãn đáp án (ground truth) bằng tay trên video.

Cách dùng:
    python tools/annotate.py --video <video> --out <gt.csv> [--step 5] [--size RỘNG CAO]

Thao tác trong cửa sổ:
- Kéo chuột: vẽ một box đang chờ gán ID.
- Phím số 0–9: gán ID cho box đang chờ.
- o: đổi mức che của box vừa gán ở frame này (0 → 1 → 2 → 0).
- n / p: sang keyframe sau / trước.
- u: bỏ box đang chờ.
- s: lưu file. q: lưu rồi thoát.

File đáp án có các cột `frame,gt_id,x,y,w,h,occluded,keyframe` (QĐ-15):
- occluded: 0 là không bị che, 1 là che một phần, 2 là che nặng (XLA 3.3).
- keyframe: 1 là box gán tay, 0 là box do máy nội suy.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.common.utils import iter_video, video_size  # noqa: E402

Box = tuple[int, int, int, int]
GtRow = tuple[int, int, int, int, int, int, int, int]
GT_HEADER = ("frame", "gt_id", "x", "y", "w", "h", "occluded", "keyframe")
OCCLUSION_LEVELS = 3
WINDOW = "Gan nhan GT"
OCCLUSION_COLORS = {0: (0, 220, 0), 1: (0, 200, 255), 2: (0, 0, 255)}


def interpolate_annotations(
    keyframes: Mapping[int, Mapping[int, Box]],
    occlusion: Mapping[int, Mapping[int, int]] | None = None,
) -> list[GtRow]:
    """Tạo đáp án cho mọi frame từ các keyframe gán tay.

    Chỉ nội suy giữa hai keyframe liên tiếp có nhãn của chính ID đó. Nếu một
    ID chỉ có một keyframe thì chỉ ghi đúng frame đó.
    """
    occlusion = occlusion or {}
    rows: set[GtRow] = set()
    ids = sorted({pid for labels in keyframes.values() for pid in labels})
    for pid in ids:
        marks = sorted((frame, labels[pid]) for frame, labels in keyframes.items() if pid in labels)
        levels = {frame: occlusion.get(frame, {}).get(pid, 0) for frame, _ in marks}
        if len(marks) == 1:
            frame, box = marks[0]
            rows.add((frame, pid, *box, levels[frame], 1))
        for (f0, b0), (f1, b1) in zip(marks, marks[1:]):
            rows.update(_segment_rows(pid, (f0, b0, levels[f0]), (f1, b1, levels[f1])))
    return sorted(rows)


def _segment_rows(pid: int, start: tuple[int, Box, int], end: tuple[int, Box, int]) -> list[GtRow]:
    """Nội suy tuyến tính một đoạn giữa hai keyframe liên tiếp của cùng một ID.

    Với t = (f − f0) / (f1 − f0):  box(f) = box(f0) + t · (box(f1) − box(f0)), rồi làm tròn.
    Frame nội suy lấy mức che lớn hơn của hai đầu đoạn, vì không biết chính xác
    lúc nào người bắt đầu bị che; chọn "che nhiều hơn" thì an toàn hơn khi chấm điểm.
    """
    f0, b0, occ0 = start
    f1, b1, occ1 = end
    rows: list[GtRow] = []
    for frame in range(f0, f1 + 1):
        t = (frame - f0) / (f1 - f0)
        box = tuple(round(a + (b - a) * t) for a, b in zip(b0, b1))
        if frame == f0:
            level, is_key = occ0, 1
        elif frame == f1:
            level, is_key = occ1, 1
        else:
            level, is_key = max(occ0, occ1), 0
        rows.append((frame, pid, *box, level, is_key))
    return rows


def write_gt(path: str | Path, rows: list[GtRow]) -> None:
    """Ghi file đáp án theo định dạng có cột occluded và keyframe."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(GT_HEADER)
        writer.writerows(rows)


def cycle_occlusion(level: int) -> int:
    """Đổi mức che theo vòng 0 → 1 → 2 → 0."""
    return (level + 1) % OCCLUSION_LEVELS


class Annotator:
    """Giữ trạng thái gán nhãn: box đang chờ, box đã gán và mức che."""

    def __init__(self) -> None:
        self.start: tuple[int, int] | None = None
        self.pending: list[Box] = []
        self.boxes: dict[int, dict[int, Box]] = {}
        self.occlusion: dict[int, dict[int, int]] = {}
        self.last_assigned: dict[int, int] = {}

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Kéo chuột từ góc này sang góc kia để tạo một box đang chờ."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            x0, y0 = self.start
            box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if box[2] > 0 and box[3] > 0:
                self.pending.append(box)
            self.start = None

    def assign_id(self, frame: int, pid: int) -> None:
        """Gán ID cho box đang chờ mới nhất; mức che bắt đầu từ 0."""
        self.boxes.setdefault(frame, {})[pid] = self.pending.pop()
        self.occlusion.setdefault(frame, {})[pid] = 0
        self.last_assigned[frame] = pid

    def toggle_occlusion(self, frame: int) -> None:
        """Đổi mức che của box vừa gán ở frame này."""
        pid = self.last_assigned.get(frame)
        if pid is None:
            return
        levels = self.occlusion.setdefault(frame, {})
        levels[pid] = cycle_occlusion(levels.get(pid, 0))

    def rows(self) -> list[GtRow]:
        """Đáp án của mọi frame sau khi nội suy."""
        return interpolate_annotations(self.boxes, self.occlusion)


def draw(frame: np.ndarray, ui: Annotator, frame_index: int) -> np.ndarray:
    """Vẽ box đã gán (màu theo mức che), box đang chờ và dòng hướng dẫn."""
    canvas = frame.copy()
    levels = ui.occlusion.get(frame_index, {})
    for pid, (x, y, w, h) in ui.boxes.get(frame_index, {}).items():
        level = levels.get(pid, 0)
        color = OCCLUSION_COLORS[level]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, f"{pid} che{level}", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for x, y, w, h in ui.pending:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)
    help_text = f"frame {frame_index} | keo chuot, so=ID, o=che, n/p/u/s/q"
    cv2.putText(canvas, help_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return canvas


def annotate_video(video: str, out: str, step: int, size: tuple[int, int]) -> None:
    """Mở cửa sổ gán nhãn trên các keyframe cách nhau `step` frame."""
    frames = [frame for _, frame in iter_video(video, size)]
    keyframe_ids = list(range(0, len(frames), step))
    ui = Annotator()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)
    position = 0
    while keyframe_ids:
        frame_index = keyframe_ids[position]
        cv2.imshow(WINDOW, draw(frames[frame_index], ui, frame_index))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("n"), ord("p")):
            delta = 1 if key == ord("n") else -1
            position = min(max(position + delta, 0), len(keyframe_ids) - 1)
            ui.pending.clear()
        elif key == ord("u") and ui.pending:
            ui.pending.pop()
        elif key == ord("o"):
            ui.toggle_occlusion(frame_index)
        elif key == ord("s"):
            write_gt(out, ui.rows())
        elif ord("0") <= key <= ord("9") and ui.pending:
            ui.assign_id(frame_index, key - ord("0"))
    cv2.destroyAllWindows()
    write_gt(out, ui.rows())


def main() -> None:
    """Đọc tham số dòng lệnh; kích thước và bước keyframe mặc định lấy từ config."""
    parser = argparse.ArgumentParser(description="Gán nhãn đáp án bằng tay")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, help="khoảng cách giữa hai keyframe (frame)")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="đổi cỡ ảnh (mặc định: giữ cỡ gốc của video, đúng với hệ thống mới)")
    args = parser.parse_args()
    size = tuple(args.size) if args.size else video_size(args.video)
    step = args.step if args.step is not None else 5
    if step <= 0:
        raise ValueError("--step phải dương")
    annotate_video(args.video, args.out, step, size)


if __name__ == "__main__":
    main()
