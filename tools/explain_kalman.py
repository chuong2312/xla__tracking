"""Ảnh minh họa cho Kalman (src/kalman.py): đoán tiếp vị trí khi người bị che.

Cách dùng:
    python -m tools.explain_kalman --scenario s2 --person 1 --out outputs/explain/kalman

Lấy đường đi thật của một người trong video tổng hợp v2, tạo "phép đo" bằng cách
cộng nhiễu ngẫu nhiên, rồi cố ý bỏ hết phép đo trong một đoạn (giả lập bị che lâu).
Ảnh trên: đường thật, các phép đo và đường Kalman ước lượng trên khung hình.
Ảnh dưới: sai lệch so với đường thật theo thời gian của phép đo và của Kalman.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.kalman import ConstantVelocityKalman  # noqa: E402
from tools.synthetic_render import make_background  # noqa: E402
from tools.synthetic_scenarios import SCENARIOS  # noqa: E402
from tools.synthetic_scene import floor_to_image, n_frames, person_box  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_NOISE_PX = 4.0


def true_centers(scenario: str, person_id: int, camera: str, fps: float) -> dict[int, np.ndarray]:
    """Tâm box đầy đủ (không bị che) của một người ở từng frame, lấy từ mô tả cảnh."""
    chosen = SCENARIOS[scenario]
    person = next(p for p in chosen.people if p.gt_id == person_id)
    H = floor_to_image(chosen.layout, camera)
    centers = {}
    for frame in range(n_frames(fps)):
        position = person.position(frame / fps)
        if position is not None:
            x, y, w, h = person_box(H, *position)
            centers[frame] = np.array([x + w / 2.0, y + h / 2.0])
    return centers


def run_filter(truth: dict[int, np.ndarray], hidden: range, cfg: dict, seed: int):
    """Chạy Kalman; trong đoạn `hidden` không có phép đo nên chỉ dự đoán."""
    rng = np.random.default_rng(seed)
    frames = sorted(truth)
    measurements = {f: truth[f] + rng.normal(0.0, MEASUREMENT_NOISE_PX, 2) for f in frames if f not in hidden}
    kalman = ConstantVelocityKalman(*measurements[frames[0]], cfg)
    estimates = {frames[0]: np.array(kalman.position)}
    for frame in frames[1:]:
        predicted = kalman.predict()
        if frame in measurements:
            estimates[frame] = np.array(kalman.correct(*measurements[frame]))
        else:
            estimates[frame] = np.array(predicted)
    return measurements, estimates


def main() -> None:
    """Vẽ ảnh minh họa Kalman cho một người."""
    parser = argparse.ArgumentParser(description="Minh họa Kalman khi người bị che")
    parser.add_argument("--scenario", default="s2")
    parser.add_argument("--person", type=int, default=1)
    parser.add_argument("--camera", default="cam_a")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hide-from", type=int, default=150, help="frame bắt đầu bị che")
    parser.add_argument("--hide-frames", type=int, default=30, help="số frame bị che")
    parser.add_argument("--out", type=Path, default=Path("outputs/explain/kalman"))
    args = parser.parse_args()
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))["kalman"]
    truth = true_centers(args.scenario, args.person, args.camera, args.fps)
    hidden = range(args.hide_from, args.hide_from + args.hide_frames)
    measurements, estimates = run_filter(truth, hidden, cfg, seed=0)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1.4, 1]})
    layout = SCENARIOS[args.scenario].layout
    background = make_background(layout, floor_to_image(layout, args.camera), np.random.default_rng(0))
    top.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    path = np.array([truth[f] for f in sorted(truth)])
    measured = np.array(list(measurements.values()))
    estimated = np.array([estimates[f] for f in sorted(estimates)])
    hidden_estimates = np.array([estimates[f] for f in hidden if f in estimates])
    top.plot(path[:, 0], path[:, 1], color="white", linewidth=3, label="đường đi thật")
    top.scatter(measured[:, 0], measured[:, 1], s=6, color="orange", label="phép đo (có nhiễu)")
    top.plot(estimated[:, 0], estimated[:, 1], color="blue", linewidth=1.5, label="Kalman ước lượng")
    if len(hidden_estimates):
        top.plot(hidden_estimates[:, 0], hidden_estimates[:, 1], color="red", linewidth=3, label="đoạn bị che: Kalman chỉ dự đoán")
    top.legend(loc="lower left", fontsize=9)
    top.set_title(f"Người {args.person}, kịch bản {args.scenario}, {args.camera}")
    top.axis("off")

    frames = sorted(estimates)
    estimate_error = [np.linalg.norm(estimates[f] - truth[f]) for f in frames]
    measure_frames = sorted(measurements)
    measure_error = [np.linalg.norm(measurements[f] - truth[f]) for f in measure_frames]
    bottom.plot(measure_frames, measure_error, ".", color="orange", markersize=3, label="sai lệch của phép đo")
    bottom.plot(frames, estimate_error, color="blue", label="sai lệch của Kalman")
    bottom.axvspan(hidden.start, hidden.stop - 1, color="red", alpha=0.15, label="đoạn bị che")
    bottom.set_xlabel("frame")
    bottom.set_ylabel("sai lệch so với vị trí thật (pixel)")
    bottom.legend(fontsize=9)
    figure.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / f"{args.scenario}_nguoi{args.person}_{args.camera}.png"
    figure.savefig(out_png, dpi=90)
    plt.close(figure)
    last_hidden = hidden.stop - 1
    print(f"Đã ghi {out_png}; cuối đoạn bị che Kalman lệch {np.linalg.norm(estimates[last_hidden] - truth[last_hidden]):.1f} px")


if __name__ == "__main__":
    main()
