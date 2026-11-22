"""Chạy detector MOG2 và xuất detections.csv tùy chọn kèm video debug."""
import argparse
import csv
import logging
from pathlib import Path
import cv2
import yaml
from legacy.config import load_config
from legacy.detector import Detector
from src.common.utils import iter_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("run_detector")

def main() -> None:
    """Đọc video đã resize, phát hiện chuyển động và ghi kết quả."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug-video")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    detector = Detector({"detector": cfg.detector, "scene": raw["scene"]})
    size = (cfg.scene.resize_width, cfg.scene.resize_height)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_writer = None
    frame_count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "x", "y", "w", "h", "area"])
        try:
            for frame_idx, frame in iter_video(args.video, size):
                boxes = detector.detect(frame)
                for x, y, width, height in boxes:
                    area = int(cv2.countNonZero(detector.get_foreground_mask()[y:y + height, x:x + width]))
                    writer.writerow([frame_idx, x, y, width, height, area])
                LOGGER.info("frame=%d detections=%d", frame_idx, len(boxes))
                if args.debug_video:
                    if debug_writer is None:
                        debug_path = Path(args.debug_video)
                        debug_path.parent.mkdir(parents=True, exist_ok=True)
                        fps_cap = cv2.VideoCapture(args.video)
                        fps = fps_cap.get(cv2.CAP_PROP_FPS)
                        fps_cap.release()
                        if fps <= 0: raise ValueError(f"Không đọc được FPS hợp lệ từ video: {args.video}")
                        debug_writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                        if not debug_writer.isOpened(): raise RuntimeError(f"Không tạo được debug video: {debug_path}")
                    debug_frame = frame.copy()
                    for x, y, width, height in boxes:
                        cv2.rectangle(debug_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    debug_writer.write(debug_frame)
                frame_count += 1
        finally:
            if debug_writer is not None: debug_writer.release()
    LOGGER.info("Đã xử lý %d frame; CSV: %s", frame_count, output)

if __name__ == "__main__": main()
