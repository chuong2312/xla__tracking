"""Đổi file hiệu chỉnh camera của EPFL thành homography sàn A → B cho hệ thống (homography.json).

Cách dùng:
    python -m tools.epfl_homography --calib data/goc/epfl_lab4p/calibration-4p.txt \
        --cam-a 0 --cam-b 1 --scene data/canh/epfl_lab4p_c0_c1 [--explain outputs/explain/homography_epfl.png]

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
from src.common.utils import camera_video


MARKERS = {"# Ground plane homography": ("ground", 3), "# Head plane homography": ("head", 3),
           "# Head plane height in camera view": ("head_line", 1)}


def _numbers_after(lines: list[str], index: int, count: int) -> list[list[float]]:
    """`count` dòng số đầu tiên sau dòng `index` (bỏ dòng trống và dòng chú thích)."""
    rows = [line for line in lines[index + 1:] if line and not line.startswith("#")]
    return [[float(v) for v in line.split()] for line in rows[:count]]


def read_calibration(path: Path) -> dict[int, dict]:
    """Đọc file hiệu chỉnh EPFL: mỗi camera có
    - "ground": ma trận 3×3 ảnh → bản đồ sàn (mặt sàn);
    - "head": ma trận 3×3 ảnh → bản đồ sàn của mặt phẳng ngang cao 1,75 m (ngang đầu người), hoặc None;
    - "head_line": khi không có "head", EPFL cho độ cao (% chiều cao ảnh, tính từ trên) của đường mà
      mặt phẳng đầu thu lại thành (camera đặt gần ngang đầu người), ngược lại None.
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result: dict[int, dict] = {}
    camera = None
    for index, line in enumerate(lines):
        if line.startswith("# Camera"):
            camera = int(line.split()[-1])
            result[camera] = {"ground": None, "head": None, "head_line": None}
        elif line in MARKERS and camera is not None:
            key, count = MARKERS[line]
            rows = _numbers_after(lines, index, count)
            result[camera][key] = rows[0][0] if key == "head_line" else np.asarray(rows, dtype=np.float64)
    return result


def read_ground_homographies(path: Path) -> dict[int, np.ndarray]:
    """Ma trận mặt sàn (ảnh → bản đồ sàn) của từng camera trong file EPFL."""
    return {camera: data["ground"] for camera, data in read_calibration(path).items()}


def to_video_pixels(g: np.ndarray, calib_size, video_size) -> np.ndarray:
    """Đổi G (ảnh → bản đồ sàn) từ tọa độ ảnh lúc hiệu chỉnh sang tọa độ ảnh của video.

    x_video = S · x_hiệu_chỉnh với S = diag(rộng_video / rộng_hc, cao_video / cao_hc, 1), nên
    G_video = G · S⁻¹. Dùng khi file hiệu chỉnh được làm trên ảnh cỡ khác video.
    """
    if calib_size is None:
        return g
    s = np.diag([video_size[0] / calib_size[0], video_size[1] / calib_size[1], 1.0])
    return g @ np.linalg.inv(s)


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


def explain(scene: Path, g_a: np.ndarray, g_b: np.ndarray, out: Path, top_view) -> None:
    """Vẽ cùng một lưới sàn (trong vùng top_view = x0, y0, rộng, cao) lên frame đầu của hai camera."""
    x0, y0, width, height = top_view
    xs, ys = np.meshgrid(np.linspace(x0, x0 + width, 10), np.linspace(y0, y0 + height, 10))
    floor_points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    hues = np.linspace(0, 170, len(floor_points)).astype(np.uint8)
    colors = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for h in hues]
    left = draw_grid(first_frame(camera_video(scene / "cam_a")), g_a, floor_points, colors)
    right = draw_grid(first_frame(camera_video(scene / "cam_b")), g_b, floor_points, colors)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(cv2.hconcat([left, right]), None, fx=2, fy=2))


def main() -> None:
    """Đọc file EPFL, ghi homography.json vào thư mục cảnh."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--cam-a", type=int, required=True)
    parser.add_argument("--cam-b", type=int, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--calib-size", type=int, nargs=2, metavar=("RONG", "CAO"),
                        help="cỡ ảnh lúc hiệu chỉnh nếu khác cỡ video (bộ 4 người: không cần)")
    parser.add_argument("--top-view", type=float, nargs=4, default=(0, 0, 360, 360), metavar=("X0", "Y0", "RONG", "CAO"),
                        help="vùng bản đồ sàn để vẽ lưới minh họa (mặc định 0 0 360 360)")
    parser.add_argument("--explain", type=Path)
    args = parser.parse_args()
    homographies = read_ground_homographies(args.calib)
    capture = cv2.VideoCapture(str(camera_video(args.scene / "cam_a")))
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    g_a = to_video_pixels(homographies[args.cam_a], args.calib_size, size)
    g_b = to_video_pixels(homographies[args.cam_b], args.calib_size, size)
    data = {"H_ab": floor_to_floor(g_a, g_b).tolist(),
            "nguon": f"{args.calib.name}, camera {args.cam_a} → camera {args.cam_b} (H_AB = G_B⁻¹ · G_A)"}
    (args.scene / "homography.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    calibration = args.calib.resolve()
    root = Path(__file__).resolve().parents[1]
    info = {"hieu_chinh": calibration.relative_to(root).as_posix() if calibration.is_relative_to(root) else str(calibration),
            "cam_a": args.cam_a, "cam_b": args.cam_b}
    (args.scene / "camera.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {args.scene / 'homography.json'} và camera.json (file hiệu chỉnh, số camera)")
    if args.explain:
        explain(args.scene, g_a, g_b, args.explain, args.top_view)
        print(f"Đã ghi {args.explain}")


if __name__ == "__main__":
    main()
