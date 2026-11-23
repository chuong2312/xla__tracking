"""Chạy hệ thống trên MỘT camera: tìm người (C) → bám người (D).

Nếu bật `stabilization`, mỗi frame được ổn định hình trước (src/detect/stabilization.py): tìm người
và bám người chạy trên frame đã ổn định, còn box ghi ra file và vẽ lên video debug được đổi về
tọa độ video gốc.
Trước đó, một lượt chạy trên các frame đầu ước lượng chiều cao người theo vị trí chân
(src/detect/height_model.py) để bước tìm người gộp đúng các mảnh của cùng một người.
Mỗi frame:
1. PersonDetector tìm box các vùng giống người, kèm mask foreground.
2. PeopleTracker cập nhật track.
3. Cập nhật nền, trừ vùng có người đang theo dõi, để người đứng yên không bị "nuốt" vào nền.
Ghi ra: detections.csv, tracks.csv và (tùy chọn) video debug.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.detect.detection import PersonDetector
from src.detect.stabilization import Stabilizer
from src.detect.height_model import HeightModel, fit_height_model, save
from src.tracking.people_tracker import PeopleTracker
from src.tracking.track import ACTIVE_STATES
from src.common.utils import iter_video

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


def draw_tracks(frame, tracks, to_original=None):
    """Vẽ box theo màu trạng thái, kèm ID, lên ảnh debug (đổi box về tọa độ gốc nếu có `to_original`)."""
    canvas = frame.copy()
    for track in tracks:
        x, y, w, h = to_original(track.box) if to_original else track.box
        color = STATE_COLORS[track.state]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, f"{track.id} {track.state}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def make_stabilizer(config: Mapping) -> Stabilizer | None:
    """Bộ ổn định hình nếu `stabilization.enabled`, ngược lại None."""
    return Stabilizer(config["stabilization"]) if config["stabilization"]["enabled"] else None


def stable_frames(video: str | Path, stabilizer: Stabilizer | None):
    """Sinh (chỉ số frame, frame gốc, frame đã ổn định, vùng hợp lệ hoặc None)."""
    for frame_index, frame in iter_video(video):
        if stabilizer is None:
            yield frame_index, frame, frame, None
        else:
            yield frame_index, frame, stabilizer.apply(frame), stabilizer.valid_mask


def estimate_heights(video: str | Path, config: Mapping) -> HeightModel | None:
    """Lượt chạy đầu: tìm người KHÔNG gộp mảnh trên các frame đầu, rồi ước lượng h(y_chân)."""
    cfg = config["detection"]["height_model"]
    detector = PersonDetector(config["detection"])
    boxes = []
    for frame_index, _, work, valid in stable_frames(video, make_stabilizer(config)):
        if frame_index >= int(cfg["frames"]):
            break
        boxes.extend(detector.detect(work, valid))
        detector.update_background(work, [])
    return fit_height_model(boxes, cfg)


def process_camera(video: str | Path, camera: str, out_dir: Path, config: Mapping, debug_video: bool = False,
                   max_frames: int | None = None) -> CameraResult:
    """Chạy C → D trên một video; trả về các dòng track (frame, id, state, x, y, w, h, n_points).

    `max_frames`: chỉ xử lý bấy nhiêu frame đầu (để chạy thử nhanh hoặc test); None = cả video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    heights = estimate_heights(video, config) if config["detection"]["height_model"]["enabled"] else None
    if heights is not None:
        save(heights, out_dir / "height_model.json")
    detector = PersonDetector(config["detection"], heights=heights)
    tracker = PeopleTracker(config["people_tracker"], config["kalman"], config["features"], heights=heights)
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
        stabilizer = make_stabilizer(config)
        original = stabilizer.to_original if stabilizer else (lambda box: tuple(box))
        for frame_index, frame, work, valid in stable_frames(video, stabilizer):
            if max_frames is not None and frame_index >= max_frames:
                break
            boxes = detector.detect(work, valid)
            detections_out.writerows([(frame_index, *original(box), box[2] * box[3]) for box in boxes])
            tracks = tracker.update(work, boxes, detector.last_mask)
            detector.update_background(work, [t.box for t in tracks if t.state in ACTIVE_STATES])
            for track in tracks:
                row = (frame_index, track.id, track.state, *original(track.box), len(track.points))
                tracks_out.writerow(row)
                result.rows.append(row)
            if writer_video is not None:
                writer_video.write(draw_tracks(frame, tracks, original))
    if writer_video is not None:
        writer_video.release()
    return result
