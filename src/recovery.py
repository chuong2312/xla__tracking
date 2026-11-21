"""Ghép lại người vừa tách ra khỏi blob gộp, dùng NHIỀU tín hiệu (XLA 2.4).

Khi hai người đi qua nhau rồi tách ra, chỉ dùng màu áo là không đủ (hai người có thể
mặc cùng màu). Chi phí ghép track t với detection d gồm ba phần:
    chi phí = w_m · min(1, khoảng cách / đường chéo)       (liên tục chuyển động, theo Kalman)
            + w_t · (1 − NCC)                             (giống ảnh mẫu chụp trước khi gộp)
            + w_c · (1 − độ giống màu)                    (màu thân trên, thân dưới)
Cặp có khoảng cách lớn hơn `recover_max_dist_ratio` × đường chéo thì bị cấm.
NCC (tương quan chéo chuẩn hóa) so hai ảnh xám cùng cỡ: 1 là giống hệt, 0 hoặc âm là không giống.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from src.color_features import body_histograms, color_similarity
from src.track import Track

TEMPLATE_SIZE = (24, 48)


def crop(frame: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    """Cắt box ra khỏi ảnh; None nếu box nằm ngoài ảnh."""
    x, y, w, h = (int(round(v)) for v in box)
    piece = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
    return piece if piece.size else None


def make_template(gray: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    """Ảnh mẫu: crop xám thu về cỡ cố định để so bằng NCC."""
    piece = crop(gray, box)
    return None if piece is None else cv2.resize(piece, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)


def ncc(template_a: np.ndarray | None, template_b: np.ndarray | None) -> float:
    """Tương quan chéo chuẩn hóa giữa hai ảnh mẫu cùng cỡ, kẹp về 0–1."""
    if template_a is None or template_b is None:
        return 0.0
    score = cv2.matchTemplate(template_a, template_b, cv2.TM_CCOEFF_NORMED)[0, 0]
    return float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0


def colors_of(frame_bgr: np.ndarray, box: Sequence[float], features_cfg: Mapping) -> dict | None:
    """Histogram màu thân trên / thân dưới của vùng box."""
    piece = crop(frame_bgr, box)
    return None if piece is None else body_histograms(piece, None, features_cfg)


def recovery_cost(track: Track, box: Sequence[float], gray: np.ndarray, frame_bgr: np.ndarray, cfg: Mapping, features_cfg: Mapping) -> float:
    """Chi phí ghép track vừa ra khỏi blob gộp với một detection; np.inf nếu bị cấm."""
    x, y, w, h = box
    distance = np.hypot(x + w / 2 - track.center[0], y + h / 2 - track.center[1]) / max(track.diagonal, 1.0)
    if distance > float(cfg["recover_max_dist_ratio"]):
        return np.inf
    template_score = ncc(track.template, make_template(gray, box))
    color_score = color_similarity(track.colors, colors_of(frame_bgr, box, features_cfg))
    return (float(cfg["recover_weight_motion"]) * min(1.0, distance)
            + float(cfg["recover_weight_template"]) * (1.0 - template_score)
            + float(cfg["recover_weight_color"]) * (1.0 - color_score))
