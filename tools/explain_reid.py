"""Ảnh minh họa cho nhận dạng lại giữa hai camera (src/reid.py), chế độ oracle.

Cách dùng:
    python -m tools.explain_reid --scene data/synthetic_v2/s4 --person 3 --out outputs/explain/nhan_dang_lai

Ảnh 1 (<kịch bản>_bang_diem.png): 5 bảng điểm — keypoint, màu, vị trí sàn, tổng chỉ ngoại hình,
tổng có vị trí sàn. Ô viền đỏ là cặp đúng theo đáp án, ô viền xanh là cặp chương trình chọn.
Ảnh 2 (<kịch bản>_nguoi<N>_khop.png): crop của cùng một người ở camera A và camera B, nối
các cặp keypoint khớp tốt (đã qua ratio test và ràng buộc dọc).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.assignment import assign  # noqa: E402
from src.color_features import apply_gains, gray_world_gains  # noqa: E402
from src.keypoint_features import extract, norm_type, normalize_crop  # noqa: E402
from src.reid import read_frames  # noqa: E402
from src.reid_similarity import good_matches  # noqa: E402
from tools.run_reid import run  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    "keypoint": "Keypoint", "mau": "Màu áo/quần", "san": "Vị trí sàn",
    "ngoai_hinh": "Tổng: chỉ ngoại hình", "ngoai_hinh_san": "Tổng: ngoại hình + sàn",
}


def plot_tables(result: dict, threshold: float, title: str, out_png: Path) -> None:
    """Vẽ 5 bảng điểm cạnh nhau."""
    ids_a, ids_b = result["ids_a"], result["ids_b"]
    figure, axes = plt.subplots(1, len(TITLES), figsize=(4 * len(TITLES), 4.2))
    for axis, (name, label) in zip(axes, TITLES.items()):
        table = result["tables"][name]
        axis.imshow(np.nan_to_num(table, nan=0.0), cmap="Greens", vmin=0, vmax=1)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                text = "—" if np.isnan(table[i, j]) else f"{table[i, j]:.2f}"
                axis.text(j, i, text, ha="center", va="center", fontsize=9)
        for i, id_a in enumerate(ids_a):
            if id_a in ids_b:
                j = ids_b.index(id_a)
                axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, color="red", linewidth=2))
        if name.startswith("ngoai_hinh"):
            for i, j in assign(1.0 - np.nan_to_num(table, nan=0.0), max_cost=1.0 - threshold):
                axis.add_patch(Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, fill=False, color="blue", linewidth=2, linestyle="--"))
        axis.set_xticks(range(len(ids_b)), [f"B{v}" for v in ids_b])
        axis.set_yticks(range(len(ids_a)), [f"A{v}" for v in ids_a])
        axis.set_title(label, fontsize=10)
    figure.suptitle(f"{title}: viền đỏ = cặp đúng theo đáp án, viền xanh đứt = cặp chương trình chọn (ngưỡng {threshold})")
    figure.tight_layout(rect=(0, 0.04, 1, 0.9))
    figure.savefig(out_png, dpi=85)
    plt.close(figure)


def best_crop(scene_dir: Path, camera: str, track, gains, cfg: dict):
    """Crop chuẩn của lần xuất hiện có box lớn nhất trong gallery của track."""
    observation = max(track.gallery, key=lambda o: o.box[2] * o.box[3])
    frame = read_frames(str(scene_dir / camera / "video.mp4"), {observation.frame})[observation.frame]
    return normalize_crop(apply_gains(frame, gains), observation.box, cfg["crop_size"])


def plot_matches(scene_dir: Path, result: dict, person: int, method: str, config: dict, out_png: Path) -> int:
    """Vẽ các cặp keypoint khớp tốt giữa crop của cùng một người ở hai camera; trả về số cặp."""
    images, features = [], []
    for camera in ("cam_a", "cam_b"):
        video = str(scene_dir / camera / "video.mp4")
        gains = gray_world_gains(read_frames(video, {0})[0])
        crop, mask = best_crop(scene_dir, camera, result["tracks"][camera][person], gains, config["features"])
        images.append(crop)
        features.append(extract(crop, mask, method, config["features"]))
    matches = good_matches(features[0], features[1], cv2.BFMatcher(norm_type(method)), config["reid"])
    keypoints = [[cv2.KeyPoint(float(x), float(y), 4) for x, y in f.points] for f in features]
    canvas = cv2.drawMatches(images[0], keypoints[0], images[1], keypoints[1], matches, None,
                             matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    canvas = cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(canvas, f"nguoi {person}: {len(matches)} cap khop tot", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_png), canvas)
    return len(matches)


def main() -> None:
    """Chạy nhận dạng lại oracle rồi vẽ hai ảnh minh họa."""
    parser = argparse.ArgumentParser(description="Minh họa nhận dạng lại giữa hai camera")
    parser.add_argument("--scene", type=Path, default=Path("data/synthetic_v2/s4"))
    parser.add_argument("--person", type=int, default=3)
    parser.add_argument("--method", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/explain/nhan_dang_lai"))
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    method = args.method or config["features"]["method"]
    result = run(args.scene, method, args.out / f"{args.scene.name}_{method}", config)
    args.out.mkdir(parents=True, exist_ok=True)
    tables_png = args.out / f"{args.scene.name}_bang_diem.png"
    plot_tables(result, float(config["reid"]["match_threshold"]), f"Kịch bản {args.scene.name}, {method}", tables_png)
    matches_png = args.out / f"{args.scene.name}_nguoi{args.person}_khop.png"
    count = plot_matches(args.scene, result, args.person, method, config, matches_png)
    print(f"Đã ghi {tables_png} và {matches_png} ({count} cặp khớp tốt)")


if __name__ == "__main__":
    main()
