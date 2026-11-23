"""Tiện ích dùng chung cho video và hình học hộp."""
from contextlib import AbstractContextManager
from pathlib import Path
from collections.abc import Iterator, Sequence
import random, time
import cv2
import numpy as np

def iter_video(path: str | Path, resize_to: tuple[int, int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Đọc video lần lượt, trả chỉ số frame và ảnh BGR."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Không mở được video: {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if resize_to is not None: frame = cv2.resize(frame, resize_to)
            yield index, frame
            index += 1
    finally: cap.release()

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def camera_video(camera_dir: str | Path) -> Path | None:
    """File video của một camera trong thư mục cảnh: <camera_dir>/video.<đuôi>.

    Nhận mọi đuôi phổ biến (mp4, avi, mov, mkv, m4v; không phân biệt hoa thường), để video gốc
    chép vào là chạy được, không phải đổi định dạng (đổi định dạng làm ảnh giảm chất lượng).
    Trả về None nếu thư mục không có video.
    """
    folder = Path(camera_dir)
    if not folder.is_dir():
        return None
    for path in sorted(folder.iterdir()):
        if path.stem == "video" and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def video_size(path: str | Path) -> tuple[int, int]:
    """(rộng, cao) gốc của video. Hệ thống mới làm việc trên ảnh gốc, nên đáp án cũng phải theo cỡ này."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Không mở được video: {path}")
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return size


def scale_rect(rect: Sequence[float], sx: float, sy: float) -> tuple[int, int, int, int]:
    """Scale box theo trục ngang/dọc và làm tròn từng thành phần."""
    if len(rect) != 4: raise ValueError("rect phải có 4 phần tử")
    return tuple(int(round(v * (sx if i % 2 == 0 else sy))) for i, v in enumerate(rect))  # type: ignore[return-value]

def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Tính IoU cho hai box (x,y,w,h), box suy biến cho IoU bằng 0."""
    if len(box_a) != 4 or len(box_b) != 4: raise ValueError("Box phải có dạng (x,y,w,h)")
    ax, ay, aw, ah = box_a; bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0: return 0.0
    iw = max(0.0, min(ax+aw,bx+bw)-max(ax,bx)); ih = max(0.0,min(ay+ah,by+bh)-max(ay,by))
    inter = iw*ih; union = aw*ah+bw*bh-inter
    return inter/union if union > 0 else 0.0

def iou_matrix(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]) -> np.ndarray:
    """Tạo ma trận IoU kích thước len(A) x len(B)."""
    return np.array([[iou(a,b) for b in boxes_b] for a in boxes_a], dtype=float).reshape((len(boxes_a),len(boxes_b)))

class Timer(AbstractContextManager["Timer"]):
    """Đo thời gian stage bằng perf_counter và cộng dồn theo tên."""
    totals: dict[str, float] = {}
    def __init__(self, stage: str): self.stage=stage; self.elapsed=0.0
    def __enter__(self) -> "Timer": self._start=time.perf_counter(); return self
    def __exit__(self, *args: object) -> None:
        self.elapsed=time.perf_counter()-self._start
        Timer.totals[self.stage]=Timer.totals.get(self.stage,0.0)+self.elapsed

def set_seed(seed: int) -> None:
    """Đặt seed cho bộ sinh số ngẫu nhiên Python và NumPy."""
    random.seed(seed); np.random.seed(seed)

def foot_point(box: Sequence[float]) -> tuple[float, float]:
    """Điểm chân của box (x, y, w, h): điểm giữa cạnh dưới, tức chỗ chân chạm sàn."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)
