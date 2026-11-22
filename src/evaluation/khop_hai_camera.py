"""Kiểm tra ghép người giữa hai camera bằng vị trí trên sàn, khi CHƯA có đáp án (QĐ-35).

Ý tưởng: cùng một người, ở cùng một frame, thì điểm chân trong ảnh A và điểm chân trong ảnh B phải rơi vào
CÙNG một chỗ trên bản đồ sàn nhìn từ trên xuống. Hiệu chỉnh EPFL cho mỗi camera một ma trận G_sàn đổi điểm
ảnh ra bản đồ sàn, nên với một cặp (track A, track B) được gán cùng global_id:
    d_f = ‖G_A · chân_A(f) − G_B · chân_B(f)‖      (f: frame cả hai track cùng có mặt)
    d = trung vị các d_f
Bản đồ sàn EPFL khoảng 1,9 cm một đơn vị (suy từ hai mặt phẳng sàn và đầu của camera 0, 1; khớp với bề rộng
người đo trên sàn khoảng 30 đơn vị), nên TAU = 40 đơn vị ≈ 0,75 m: xa hơn thế thì không thể là một người.
Mỗi cặp được xếp vào một trong ba loại:
- khop: có ít nhất MIN_FRAMES frame chung và d ≤ TAU;
- lech: có ít nhất MIN_FRAMES frame chung và d > TAU, tức gần như chắc chắn ghép nhầm hai người;
- khong_kiem_duoc: quá ít frame chung (track này xuất hiện khi track kia đã mất), không nói được gì.
Ngoài ra đếm các "cặp hình học": cặp track A, B có ít nhất MIN_PAIR_FRAMES frame chung với d ≤ TAU (gần như
chắc chắn cùng một người), và xem bao nhiêu cặp trong số đó được bước nhận dạng lại gán cùng global_id.

Đây là kiểm tra CHÉO bằng hình học camera do EPFL đo, không phải đáp án, và không học gì từ dữ liệu. Nó chỉ
độc lập với biến thể "ngoai_hinh" (chỉ dùng ngoại hình). Biến thể "ngoai_hinh_san" đã dùng chính vị trí sàn
để ghép, nên điểm của nó cao hơn là đương nhiên (QĐ-02). Các ngưỡng là định nghĩa của thước đo nên cố định
trong code, không đưa vào config để không ai chỉnh thước đo cho đẹp số.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from src.common.homography import project

RELIABLE = ("confirmed", "static")  # không lấy "merged": khung đang dính người khác, chân không tin được
TAU = 40.0  # đơn vị bản đồ sàn EPFL, ≈ 0,75 m
MIN_FRAMES = 10  # frame chung tối thiểu để phán một cặp đã ghép
MIN_PAIR_FRAMES = 25  # frame chung tối thiểu (1 giây) để coi là "cặp hình học"
BORDER_PX = 2  # khung chạm mép dưới ảnh trong chừng này pixel thì chân bị cắt, bỏ qua


def floor_tracks(rows: list[dict], ground: np.ndarray, image_height: int) -> dict[int, dict[int, np.ndarray]]:
    """Vị trí chân trên bản đồ sàn của từng track: {track_id: {frame: (X, Y)}}; bỏ khung bị cắt chân."""
    feet = defaultdict(dict)
    for row in rows:
        x, y, w, h = (int(row[k]) for k in "xywh")
        if row["state"] in RELIABLE and y + h < image_height - BORDER_PX:
            feet[int(row["track_id"])][int(row["frame"])] = (x + w / 2.0, y + h)
    result = {}
    for track_id, points in feet.items():
        frames = sorted(points)
        floor = project(ground, [points[f] for f in frames])
        result[track_id] = dict(zip(frames, floor))
    return result


def pair_distance(a: Mapping[int, np.ndarray], b: Mapping[int, np.ndarray]) -> tuple[float | None, int]:
    """(trung vị khoảng cách chân trên sàn, số frame chung) của hai track; (None, 0) nếu không có frame chung."""
    common = sorted(set(a) & set(b))
    if not common:
        return None, 0
    distances = [float(np.linalg.norm(a[f] - b[f])) for f in common]
    return float(np.median(distances)), len(common)


def judge(distance: float | None, count: int) -> str:
    """Xếp một cặp đã ghép vào khop / lech / khong_kiem_duoc."""
    if count < MIN_FRAMES:
        return "khong_kiem_duoc"
    return "khop" if distance <= TAU else "lech"


def geometric_pairs(tracks_a: Mapping, tracks_b: Mapping) -> set[tuple[int, int]]:
    """Các cặp (track A, track B) gần như chắc chắn là cùng một người theo vị trí sàn."""
    pairs = set()
    for id_a, a in tracks_a.items():
        for id_b, b in tracks_b.items():
            distance, count = pair_distance(a, b)
            if count >= MIN_PAIR_FRAMES and distance <= TAU:
                pairs.add((id_a, id_b))
    return pairs


def read_global_pairs(path: Path) -> list[tuple[int, int]]:
    """Các cặp (track A, track B) cùng global_id trong global_ids_<biến thể>.csv (bỏ global_id = −1)."""
    members = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["global_id"] != "-1":
                members[row["global_id"]][row["cam"]] = int(row["track_id"])
    return [(m["cam_a"], m["cam_b"]) for m in members.values() if "cam_a" in m and "cam_b" in m]


def check_pairs(matched: list[tuple[int, int]], tracks_a: Mapping, tracks_b: Mapping, sure: set) -> dict:
    """Đếm khop / lech / khong_kiem_duoc cho các cặp đã ghép, và số cặp hình học được ghép đúng."""
    counts = {"ghep": len(matched), "khop": 0, "lech": 0, "khong_kiem_duoc": 0}
    for id_a, id_b in matched:
        distance, count = pair_distance(tracks_a.get(id_a, {}), tracks_b.get(id_b, {}))
        counts[judge(distance, count)] += 1
    counts["cap_hinh_hoc"] = len(sure)
    counts["bat_duoc"] = len(sure & set(matched))
    return counts


def cross_camera_check(run_dir: Path, grounds: Mapping[str, np.ndarray], heights: Mapping[str, int], variants) -> dict:
    """Kết quả kiểm tra cho từng biến thể nhận dạng lại có file global_ids_<biến thể>.csv trong run_dir."""
    tracks = {}
    for camera in ("cam_a", "cam_b"):
        with (run_dir / camera / "tracks_linked.csv").open(encoding="utf-8") as stream:
            tracks[camera] = floor_tracks(list(csv.DictReader(stream)), grounds[camera], heights[camera])
    sure = geometric_pairs(tracks["cam_a"], tracks["cam_b"])
    result = {}
    for variant in variants:
        path = run_dir / f"global_ids_{variant}.csv"
        if path.exists():
            result[variant] = check_pairs(read_global_pairs(path), tracks["cam_a"], tracks["cam_b"], sure)
    return result
