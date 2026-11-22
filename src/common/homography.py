"""Homography mặt sàn: chuyển một điểm trên sàn từ ảnh camera này sang ảnh camera kia.

Homography là một ma trận H cỡ 3×3. Với điểm (x, y) ở ảnh A:
    [u, v, w]ᵀ = H · [x, y, 1]ᵀ,   x_B = u / w,   y_B = v / w
Phép biến đổi này chỉ đúng cho các điểm cùng nằm trên một mặt phẳng, ở đây là mặt
sàn. Vì vậy chỉ dùng nó cho điểm chân người, và chỉ coi là thông tin bổ trợ: khi sàn
không phẳng, chân bị che hoặc hai video lệch giờ thì kết quả có thể sai (QĐ-02).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

MIN_POINTS = 4


def project(H: np.ndarray, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Chiếu N điểm (mảng N×2) qua homography H; trả về mảng N×2."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def estimate(points_a, points_b, ransac_threshold_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Ước lượng H sao cho điểm ở ảnh A, sau khi chiếu, trùng điểm tương ứng ở ảnh B.

    - Đúng 4 cặp điểm: giải trực tiếp, vì H có 8 ẩn và mỗi cặp điểm cho 2 phương trình.
    - Hơn 4 cặp: dùng RANSAC để loại cặp click nhầm. Cặp nào chiếu lệch quá
      `ransac_threshold_px` pixel bị coi là cặp sai.
    Trả về (H, errors). errors[i] là khoảng cách (pixel) giữa điểm A thứ i sau khi
    chiếu và điểm B thứ i, gọi là "sai số chiếu lại". Càng nhỏ càng tốt.
    """
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if len(a) != len(b):
        raise ValueError(f"Số điểm ở hai camera phải bằng nhau: {len(a)} và {len(b)}")
    if len(a) < MIN_POINTS:
        raise ValueError(f"Cần ít nhất {MIN_POINTS} cặp điểm, đang có {len(a)}")
    method = 0 if len(a) == MIN_POINTS else cv2.RANSAC
    H, _ = cv2.findHomography(a, b, method, ransac_threshold_px)
    if H is None:
        raise ValueError("Không tính được homography: các điểm có thể gần thẳng hàng")
    errors = np.linalg.norm(project(H, a) - b, axis=1)
    return H, errors


def save(path: str | Path, H: np.ndarray, points_a, points_b, image_size: Sequence[int], errors: np.ndarray) -> None:
    """Lưu kết quả ra JSON để các bước sau (Re-ID) đọc lại."""
    data = {
        "H_ab": np.asarray(H).tolist(),
        "points_a": np.asarray(points_a).tolist(),
        "points_b": np.asarray(points_b).tolist(),
        "image_size": list(image_size),
        "reprojection_errors_px": np.asarray(errors).tolist(),
        "mean_error_px": float(np.mean(errors)),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def scene_homography(scene_dir: str | Path) -> np.ndarray | None:
    """Homography sàn A → B của một cảnh (file homography.json trong thư mục cảnh); None nếu không có."""
    path = Path(scene_dir) / "homography.json"
    return load(path)["H_ab"] if path.exists() else None


def load(path: str | Path) -> dict:
    """Đọc file JSON đã lưu; trường H_ab được đổi lại thành mảng numpy 3×3."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["H_ab"] = np.asarray(data["H_ab"], dtype=np.float64)
    return data
