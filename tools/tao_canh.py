"""Tạo thư mục cảnh hai camera cho run.py từ hai file video bất kỳ.

Cách dùng (chạy được cả trên Windows lẫn WSL/Linux):
    python -m tools.tao_canh --cam-a data/real/epfl/4p-c0.mp4 --cam-b data/real/epfl/4p-c1.mp4 --out data/real/epfl_4p_c0_c1

run.py cần cấu trúc <cảnh>/cam_a/video.mp4 và <cảnh>/cam_b/video.mp4. Công cụ này CHÉP video vào đúng
chỗ (không dùng liên kết tắt, vì liên kết tạo trong WSL không mở được trên Windows).
- Video .mp4: chép nguyên file.
- Video định dạng khác (.avi …): đọc từng frame bằng OpenCV rồi ghi lại thành .mp4 (giữ nguyên fps, kích thước).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


def reencode(source: Path, target: Path) -> int:
    """Ghi lại video sang .mp4 (codec mp4v); trả về số frame đã ghi."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Không mở được video {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        count += 1
    capture.release()
    writer.release()
    return count


def place(source: Path, camera_dir: Path) -> None:
    """Đặt video vào <camera_dir>/video.mp4: chép nếu đã là .mp4, ngược lại ghi lại."""
    if not source.is_file():
        raise SystemExit(f"Không thấy file video {source}")
    camera_dir.mkdir(parents=True, exist_ok=True)
    target = camera_dir / "video.mp4"
    if target.is_symlink():
        target.unlink()
    if source.suffix.lower() == ".mp4":
        shutil.copyfile(source, target)
        print(f"Đã chép {source} → {target}")
    else:
        frames = reencode(source, target)
        print(f"Đã ghi lại {source} → {target} ({frames} frame)")


def main() -> None:
    """Đọc tham số và tạo thư mục cảnh."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-a", type=Path, required=True, help="video của camera A")
    parser.add_argument("--cam-b", type=Path, required=True, help="video của camera B")
    parser.add_argument("--out", type=Path, required=True, help="thư mục cảnh, ví dụ data/real/epfl_4p_c0_c1")
    args = parser.parse_args()
    place(args.cam_a, args.out / "cam_a")
    place(args.cam_b, args.out / "cam_b")
    print(f"Xong. Chạy: python run.py --scene {args.out}")


if __name__ == "__main__":
    main()
