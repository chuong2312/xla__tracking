"""Click các điểm băng keo trên sàn ở hai camera để tính homography mặt sàn A → B.

Cách dùng:
    python tools/calibrate_floor.py --cam-a <video_a> --cam-b <video_b> --out <homography.json>
    python tools/calibrate_floor.py --cam-a <video_a> --cam-b <video_b> --out <homography.json> --explain <anh.png>

Cửa sổ hiện ảnh cam_a (bên trái) và cam_b (bên phải), lấy ở frame đầu (cảnh trống).
- Click vào nửa trái để thêm điểm ở cam_a, nửa phải để thêm điểm ở cam_b.
  Phải click các điểm theo CÙNG MỘT THỨ TỰ ở hai camera.
- u: xóa điểm vừa click. s: lưu (cần ít nhất 4 cặp). q: thoát.
Với --explain: không mở cửa sổ, chỉ đọc file đã lưu và xuất ảnh kiểm tra.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.common import homography  # noqa: E402

WINDOW = "Click diem san: trai=cam_a, phai=cam_b"


def first_frame(video: str, frame_index: int, size: tuple[int, int] | None) -> np.ndarray:
    """Đọc một frame (mặc định frame 0, lúc cảnh còn trống)."""
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Không đọc được frame {frame_index} của {video}")
    return cv2.resize(frame, size) if size else frame


def draw_points(image: np.ndarray, points, color: tuple[int, int, int], label_prefix: str = "") -> None:
    """Vẽ các điểm kèm số thứ tự để dễ click đúng thứ tự ở camera kia."""
    for index, (x, y) in enumerate(points, start=1):
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, 5, color, -1)
        cv2.putText(image, f"{label_prefix}{index}", (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def render_check(frame_b: np.ndarray, H: np.ndarray, points_a, points_b, errors) -> np.ndarray:
    """Ảnh kiểm tra trên cam_b: điểm click (xanh lá), điểm cam_a chiếu sang (đỏ), đa giác chiếu sang (đỏ).

    Nếu homography đúng thì điểm đỏ nằm gần như trùng điểm xanh.
    """
    canvas = frame_b.copy()
    projected = homography.project(H, points_a)
    polygon = np.round(projected).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polygon], True, (0, 0, 255), 1)
    draw_points(canvas, points_b, (0, 200, 0))
    draw_points(canvas, projected, (0, 0, 255), "A")
    text = f"sai so chieu lai TB = {float(np.mean(errors)):.2f} px"
    cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def click_points(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[list, list] | None:
    """Mở cửa sổ cho người dùng click; trả về hai danh sách điểm, hoặc None nếu thoát."""
    width = frame_a.shape[1]
    points = {"a": [], "b": []}
    order: list[str] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        side = "a" if x < width else "b"
        points[side].append((x if side == "a" else x - width, y))
        order.append(side)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        left = frame_a.copy()
        right = frame_b.copy()
        draw_points(left, points["a"], (0, 200, 0))
        draw_points(right, points["b"], (0, 200, 0))
        cv2.imshow(WINDOW, np.hstack([left, right]))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and order:
            points[order.pop()].pop()
        elif key == ord("s") and len(points["a"]) == len(points["b"]) >= homography.MIN_POINTS:
            cv2.destroyAllWindows()
            return points["a"], points["b"]
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None


def main() -> None:
    """Click điểm và lưu homography, hoặc xuất ảnh kiểm tra với --explain."""
    parser = argparse.ArgumentParser(description="Tính homography mặt sàn giữa hai camera")
    parser.add_argument("--cam-a", required=True)
    parser.add_argument("--cam-b", required=True)
    parser.add_argument("--out", required=True, help="file JSON lưu homography")
    parser.add_argument("--frame", type=int, default=0, help="frame dùng để click (nên là cảnh trống)")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--explain", help="đường dẫn ảnh PNG kiểm tra; không mở cửa sổ click")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    size = tuple(args.size) if args.size else None
    frame_a = first_frame(args.cam_a, args.frame, size)
    frame_b = first_frame(args.cam_b, args.frame, size)
    if args.explain:
        saved = homography.load(args.out)
        image = render_check(frame_b, saved["H_ab"], saved["points_a"], saved["points_b"], saved["reprojection_errors_px"])
        Path(args.explain).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.explain, image)
        print(f"Đã ghi ảnh kiểm tra: {args.explain}")
        return
    clicked = click_points(frame_a, frame_b)
    if clicked is None:
        print("Đã thoát, không lưu.")
        return
    threshold = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["homography"]["ransac_threshold_px"]
    H, errors = homography.estimate(clicked[0], clicked[1], threshold)
    homography.save(args.out, H, clicked[0], clicked[1], frame_a.shape[1::-1], errors)
    print(f"Đã lưu {args.out}; sai số chiếu lại trung bình {float(np.mean(errors)):.2f} px")


if __name__ == "__main__":
    main()
