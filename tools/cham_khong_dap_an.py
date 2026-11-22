"""Chấm kết quả run.py khi cảnh CHƯA có đáp án (src/evaluation/khong_dap_an.py, QĐ-34).

Cách dùng:
    python run.py --scene data/canh/epfl_lab4p_c0_c1
    python -m tools.cham_khong_dap_an --run outputs/he_thong/epfl_lab4p_c0_c1 --scene data/canh/epfl_lab4p_c0_c1

In và ghi `khong_dap_an.json` vào thư mục kết quả: so_id, track_moi_frame, phu_nguoi, track_ma, nhay, cao_hop_ly
cho từng camera. Nếu cảnh có `camera.json` (do tools/epfl_homography.py ghi: file hiệu chỉnh và số camera) thì
dùng hiệu chỉnh để biết khung nào cao đúng cỡ người.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.utils import camera_video, video_size
from src.evaluation.khong_dap_an import scorecard
from tools.epfl_homography import read_calibration

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam_a", "cam_b")


def camera_calibrations(scene: Path) -> dict[str, dict | None]:
    """Hiệu chỉnh của cam_a, cam_b theo camera.json của cảnh; None nếu cảnh không có."""
    info_path = scene / "camera.json"
    if not info_path.exists():
        return {camera: None for camera in CAMERAS}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    calibration = read_calibration(ROOT / info["hieu_chinh"])
    return {camera: calibration[info[camera]] for camera in CAMERAS if camera in info}


def score_scene(run_dir: Path, scene: Path) -> dict:
    """Bảng chỉ số cho mọi camera có kết quả trong run_dir."""
    calibrations = camera_calibrations(scene)
    result = {}
    for camera in CAMERAS:
        if (run_dir / camera / "tracks_linked.csv").exists() and camera_video(scene / camera) is not None:
            height = video_size(camera_video(scene / camera))[1]
            result[camera] = scorecard(run_dir / camera, calibrations.get(camera), height)
    return result


def main() -> None:
    """Chấm một thư mục kết quả và ghi khong_dap_an.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="thư mục kết quả run.py, ví dụ outputs/he_thong/<cảnh>")
    parser.add_argument("--scene", type=Path, required=True, help="thư mục cảnh, ví dụ data/canh/<cảnh>")
    args = parser.parse_args()
    result = score_scene(args.run, args.scene)
    (args.run / "khong_dap_an.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for camera, values in result.items():
        print(camera, "  ".join(f"{k}={v}" for k, v in values.items()))


if __name__ == "__main__":
    main()
