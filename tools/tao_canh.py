"""Tạo thư mục cảnh cho run.py từ các file video gốc (chạy được cả trên Windows lẫn WSL/Linux).

Cách dùng:
    python -m tools.tao_canh --cam-a data/goc/epfl_lab4p/4p-c0.avi --cam-b data/goc/epfl_lab4p/4p-c1.avi \
        --out data/canh/epfl_lab4p_c0_c1
    python -m tools.tao_canh --cam-a duong/dan/video_mot_camera.mp4 --out data/canh/ten_canh   # chỉ một camera

run.py cần cấu trúc <cảnh>/cam_a/video.<đuôi> (và cam_b/video.<đuôi> nếu có camera thứ hai), xem
data/README.md. Công cụ CHÉP nguyên file (giữ đuôi gốc: mp4, avi, mov…), không mã hóa lại nên ảnh
không giảm chất lượng, và không dùng liên kết tắt (liên kết tạo trong WSL không mở được trên Windows).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.common.utils import VIDEO_EXTENSIONS, camera_video


def place(source: Path, camera_dir: Path) -> Path:
    """Chép `source` thành <camera_dir>/video.<đuôi gốc>; xóa video cũ của camera đó nếu có."""
    if not source.is_file():
        raise SystemExit(f"Không thấy file video {source}")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit(f"Đuôi {source.suffix} chưa hỗ trợ; dùng một trong {', '.join(VIDEO_EXTENSIONS)}")
    camera_dir.mkdir(parents=True, exist_ok=True)
    old = camera_video(camera_dir)
    while old is not None:
        old.unlink()
        old = camera_video(camera_dir)
    target = camera_dir / f"video{source.suffix.lower()}"
    shutil.copyfile(source, target)
    print(f"Đã chép {source} → {target}")
    return target


def main() -> None:
    """Đọc tham số và tạo thư mục cảnh."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-a", type=Path, required=True, help="video của camera A")
    parser.add_argument("--cam-b", type=Path, help="video của camera B (bỏ trống nếu chỉ có một camera)")
    parser.add_argument("--out", type=Path, required=True, help="thư mục cảnh, ví dụ data/canh/ten_canh")
    args = parser.parse_args()
    place(args.cam_a, args.out / "cam_a")
    if args.cam_b is not None:
        place(args.cam_b, args.out / "cam_b")
    print(f"Xong. Chạy: python run.py --scene {args.out}")


if __name__ == "__main__":
    main()
