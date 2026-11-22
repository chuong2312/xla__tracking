"""Đổi file hiệu chỉnh camera của EPFL thành homography sàn A → B cho hệ thống (homography.json).

Cách dùng:
    python -m tools.epfl_homography --calib data/real/epfl/calibration-4p.txt \
        --cam-a 0 --cam-b 1 --scene data/real/epfl_4p_c0_c1 [--explain outputs/explain/homography_epfl.png]

File của EPFL cho mỗi camera một ma trận G "ảnh → bản đồ sàn nhìn từ trên xuống":
    [X, Y, W]ᵀ = G · [x, y, 1]ᵀ
Điểm chân ở ảnh A → bản đồ sàn bằng G_A, rồi bản đồ sàn → ảnh B bằng nghịch đảo G_B:
    H_AB = G_B⁻¹ · G_A
Không cần click tay như tools/calibrate_floor.py, vì EPFL đã đo sẵn.
Ảnh --explain vẽ cùng một lưới ô vuông trên sàn lên cả hai camera (cùng màu = cùng điểm sàn):
nếu ma trận đúng thì lưới nằm phẳng trên sàn và các điểm cùng màu là cùng một chỗ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.common.homography import project


def read_ground_homographies(path: Path) -> dict[int, np.ndarray]:
    """Đọc ma trận 'Ground plane homography' của từng camera trong file EPFL."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result: dict[int, np.ndarray] = {}
    camera = None
    for index, line in enumerate(lines):
        if line.startswith("# Camera"):
            camera = int(line.split()[-1])
        if line == "# Ground plane homography" and camera is not None:
            rows = [[float(v) for v in lines[index + k].split()] for k in (1, 2, 3)]
            result[camera] = np.asarray(rows, dtype=np.float64)
    return result


def floor_to_floor(g_a: np.ndarray, g_b: np.ndarray) -> np.ndarray:
    """H_AB = G_B⁻¹ · G_A, chuẩn hóa để phần tử cuối bằng 1."""
    h = np.linalg.inv(g_b) @ g_a
    return h / h[2, 2]


def first_frame(video: Path) -> np.ndarray:
    """Frame đầu của video, dùng làm nền cho ảnh minh họa."""
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Không đọc được {video}")
    return frame


def draw_grid(frame: np.ndarray, g: np.ndarray, floor_points: np.ndarray, colors) -> np.ndarray:
    """Chiếu các điểm lưới sàn vào ảnh bằng G⁻¹ rồi vẽ chấm màu."""
    canvas = frame.copy()
    for (x, y), color in zip(project(np.linalg.inv(g), floor_points), colors):
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
    return canvas


def explain(scene: Path, g_a: np.ndarray, g_b: np.ndarray, out: Path) -> None:
    """Vẽ cùng một lưới sàn lên frame đầu của hai camera, đặt cạnh nhau."""
    xs, ys = np.meshgrid(np.linspace(0, 360, 10), np.linspace(0, 360, 10))
    floor_points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    hues = np.linspace(0, 170, len(floor_points)).astype(np.uint8)
    colors = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for h in hues]
    left = draw_grid(first_frame(scene / "cam_a/video.mp4"), g_a, floor_points, colors)
    right = draw_grid(first_frame(scene / "cam_b/video.mp4"), g_b, floor_points, colors)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(cv2.hconcat([left, right]), None, fx=2, fy=2))


def main() -> None:
    """Đọc file EPFL, ghi homography.json vào thư mục cảnh."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--cam-a", type=int, required=True)
    parser.add_argument("--cam-b", type=int, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--explain", type=Path)
    args = parser.parse_args()
    homographies = read_ground_homographies(args.calib)
    g_a, g_b = homographies[args.cam_a], homographies[args.cam_b]
    data = {"H_ab": floor_to_floor(g_a, g_b).tolist(),
            "nguon": f"{args.calib.name}, camera {args.cam_a} → camera {args.cam_b} (H_AB = G_B⁻¹ · G_A)"}
    (args.scene / "homography.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {args.scene / 'homography.json'}")
    if args.explain:
        explain(args.scene, g_a, g_b, args.explain)
        print(f"Đã ghi {args.explain}")


if __name__ == "__main__":
    main()
