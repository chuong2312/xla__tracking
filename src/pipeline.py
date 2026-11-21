"""Chạy hệ thống trên MỘT camera: tìm người (C) → bám người (D) → đếm qua cửa (E).

Mỗi frame:
1. PersonDetector tìm box các vùng giống người, kèm mask foreground.
2. PeopleTracker cập nhật track.
3. Cập nhật nền, trừ vùng có người đang theo dõi (để người đứng yên không bị nuốt vào nền).
4. DoorCounter nhận điểm chân của các track đang hoạt động. Track nào vừa kết thúc thì được
   xóa khỏi bộ đếm SAU khi đã cập nhật lần cuối (XLA 2.17).
Ghi ra: detections.csv, tracks.csv và (tùy chọn) video debug.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.detection import PersonDetector
from src.door import DoorCounter, DoorEvent, foot_point
from src.people_tracker import PeopleTracker
from src.track import ACTIVE_STATES
from src.utils import iter_video

STATE_COLORS = {"tentative": (0, 200, 255), "confirmed": (0, 220, 0), "merged": (255, 120, 0), "static": (255, 0, 255), "lost": (0, 0, 255)}


@dataclass
class CameraResult:
    """Kết quả của một camera."""

    camera: str
    fps: float
    image_size: tuple[int, int]
    tracks_csv: Path
    detections_csv: Path
    rows: list[tuple] = field(default_factory=list)
    events: list[DoorEvent] = field(default_factory=list)


def _draw(frame, tracks, door_line, events_so_far: int):
    """Vẽ box theo màu trạng thái, ID, vạch cửa và số lượt qua cửa lên ảnh debug."""
    canvas = frame.copy()
    for track in tracks:
        x, y, w, h = track.box
        color = STATE_COLORS[track.state]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, f"{track.id} {track.state}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if door_line is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in door_line)
        cv2.line(canvas, (x1, y1), (x2, y2), (30, 60, 140), 2)
        cv2.putText(canvas, f"luot qua cua: {events_so_far}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def process_camera(video: str | Path, camera: str, out_dir: Path, config: Mapping,
                   door_line: Sequence[float] | None = None, inside_side: str | None = None,
                   debug_video: bool = False) -> CameraResult:
    """Chạy C → D → E trên một video; trả về các dòng track và sự kiện qua cửa."""
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    detector = PersonDetector(config["detection"])
    tracker = PeopleTracker(config["people_tracker"], config["kalman"], config["features"])
    door = None
    if door_line is not None:
        door = DoorCounter(door_line, config["events"]["crossing_margin"], inside_side or config["events"]["inside_side"])
    result = CameraResult(camera, fps, size, out_dir / "tracks.csv", out_dir / "detections.csv")
    writer_video = None
    if debug_video:
        writer_video = cv2.VideoWriter(str(out_dir / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    with result.tracks_csv.open("w", newline="", encoding="utf-8") as tracks_file, \
            result.detections_csv.open("w", newline="", encoding="utf-8") as detections_file:
        tracks_out = csv.writer(tracks_file)
        tracks_out.writerow(["frame", "track_id", "state", "x", "y", "w", "h", "n_points"])
        detections_out = csv.writer(detections_file)
        detections_out.writerow(["frame", "x", "y", "w", "h", "area"])
        for frame_index, frame in iter_video(video):
            boxes = detector.detect(frame)
            detections_out.writerows([(frame_index, *box, box[2] * box[3]) for box in boxes])
            tracks = tracker.update(frame, boxes, detector.last_mask)
            detector.update_background(frame, [t.box for t in tracks if t.state in ACTIVE_STATES])
            for track in tracks:
                row = (frame_index, track.id, track.state, *track.box, len(track.points))
                tracks_out.writerow(row)
                result.rows.append(row)
                if door is not None and track.state in ACTIVE_STATES:
                    door.update(frame_index, track.id, foot_point(track.box))
                if door is not None and track.state == "lost":
                    door.finish_track(track.id)
            if writer_video is not None:
                writer_video.write(_draw(frame, tracks, door_line, len(door.events) if door else 0))
    if writer_video is not None:
        writer_video.release()
    result.events = list(door.events) if door else []
    return result
