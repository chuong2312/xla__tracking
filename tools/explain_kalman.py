"""Ảnh minh họa cho Kalman (src/common/kalman.py) trên video thật: đoán tiếp vị trí khi người bị che.

Cách dùng (chạy hệ thống trước để có tracks_linked.csv):
    python run.py --scene data/canh/epfl_lab4p_c0_c1
    python -m tools.explain_kalman --run outputs/he_thong/epfl_lab4p_c0_c1 --scene data/canh/epfl_lab4p_c0_c1 --camera cam_a

Lấy đoạn liền mạch dài nhất mà một người được bám ở trạng thái confirmed. Tâm khung ở từng frame là
"vị trí đo được" trên video thật. Cố ý giấu các phép đo trong một đoạn (giả như người bị che) rồi chạy
Kalman: ngoài đoạn giấu, Kalman được sửa theo phép đo; trong đoạn giấu, Kalman chỉ dự đoán theo vận tốc.
- Ảnh trên: đường đo được, đường Kalman, đoạn bị giấu, vẽ trên một frame thật.
- Ảnh dưới: Kalman lệch bao nhiêu pixel so với vị trí đo được, theo từng frame.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.kalman import ConstantVelocityKalman  # noqa: E402
from src.common.utils import camera_video  # noqa: E402
from src.reid.reid import read_frames  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def longest_run(tracks_csv: Path) -> tuple[int, dict[int, np.ndarray]]:
    """(track id, {frame: tâm khung}) của đoạn confirmed liền mạch dài nhất."""
    by_track: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with tracks_csv.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["state"] == "confirmed":
                x, y, w, h = (float(row[k]) for k in "xywh")
                by_track[int(row["track_id"])][int(row["frame"])] = np.array([x + w / 2.0, y + h / 2.0])
    best: tuple[int, dict[int, np.ndarray]] = (-1, {})
    for track_id, centers in by_track.items():
        frames = sorted(centers)
        start = 0
        for i in range(1, len(frames) + 1):
            if i == len(frames) or frames[i] != frames[i - 1] + 1:
                if i - start > len(best[1]):
                    best = (track_id, {f: centers[f] for f in frames[start:i]})
                start = i
    return best


def run_filter(measured: dict[int, np.ndarray], hidden: range, cfg: dict) -> dict[int, np.ndarray]:
    """Kalman trên đoạn đo được; trong đoạn `hidden` không có phép đo nên chỉ dự đoán."""
    frames = sorted(measured)
    kalman = ConstantVelocityKalman(*measured[frames[0]], cfg)
    estimates = {frames[0]: np.array(kalman.position)}
    for frame in frames[1:]:
        predicted = kalman.predict()
        estimates[frame] = np.array(predicted if frame in hidden else kalman.correct(*measured[frame]))
    return estimates


def plot(frame_image: np.ndarray, measured, estimates, hidden: range, title: str, out_png: Path) -> None:
    """Hai ảnh: đường đi trên frame thật, và sai lệch theo thời gian."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 10), gridspec_kw={"height_ratios": [1.3, 1]})
    top.imshow(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
    path = np.array([measured[f] for f in sorted(measured)])
    estimated = np.array([estimates[f] for f in sorted(estimates)])
    gap = np.array([estimates[f] for f in hidden if f in estimates])
    top.plot(path[:, 0], path[:, 1], color="white", linewidth=3, label="vị trí đo được (video thật)")
    top.plot(estimated[:, 0], estimated[:, 1], color="blue", linewidth=1.5, label="Kalman ước lượng")
    top.plot(gap[:, 0], gap[:, 1], color="red", linewidth=3, label="đoạn giấu phép đo: Kalman chỉ dự đoán")
    top.legend(loc="lower left", fontsize=9)
    top.set_title(title)
    top.axis("off")
    frames = sorted(estimates)
    bottom.plot(frames, [np.linalg.norm(estimates[f] - measured[f]) for f in frames], color="blue")
    bottom.axvspan(hidden.start, hidden.stop - 1, color="red", alpha=0.15, label="đoạn giấu phép đo")
    bottom.set_xlabel("frame")
    bottom.set_ylabel("Kalman lệch so với vị trí đo được (pixel)")
    bottom.legend(fontsize=9)
    figure.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=90)
    plt.close(figure)


def main() -> None:
    """Vẽ ảnh minh họa Kalman cho đoạn bám dài nhất của một camera."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("outputs/he_thong/epfl_lab4p_c0_c1"), help="thư mục kết quả run.py")
    parser.add_argument("--scene", type=Path, default=Path("data/canh/epfl_lab4p_c0_c1"))
    parser.add_argument("--camera", default="cam_a")
    parser.add_argument("--hide-frames", type=int, default=30, help="số frame giấu phép đo (25 frame = 1 giây)")
    parser.add_argument("--out", type=Path, default=Path("outputs/explain/kalman"))
    args = parser.parse_args()
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))["kalman"]
    track_id, measured = longest_run(args.run / args.camera / "tracks_linked.csv")
    if len(measured) < args.hide_frames + 20:
        raise SystemExit("Không có đoạn bám đủ dài; chạy run.py trước hoặc giảm --hide-frames.")
    frames = sorted(measured)
    start = frames[len(frames) // 2]
    hidden = range(start, start + args.hide_frames)
    estimates = run_filter(measured, hidden, cfg)
    image = read_frames(str(camera_video(args.scene / args.camera)), {start})[start]
    out_png = args.out / f"{args.scene.name}_{args.camera}_track{track_id}.png"
    plot(image, measured, estimates, hidden, f"Track {track_id}, {args.scene.name} {args.camera}", out_png)
    last = hidden.stop - 1
    print(f"Đã ghi {out_png}; cuối đoạn giấu {args.hide_frames} frame, Kalman lệch {np.linalg.norm(estimates[last] - measured[last]):.1f} px")


if __name__ == "__main__":
    main()
