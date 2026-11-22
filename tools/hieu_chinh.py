"""Hiệu chỉnh tham số: chạy toàn hệ thống với nhiều giá trị tham số, in bảng điểm để so sánh.

CHỈ DÙNG VIDEO THẬT (QĐ-31): mọi cảnh trong data/canh/ (mỗi cảnh là một thư mục có cam_a/video.*).

Cách dùng:
    python -m tools.hieu_chinh                                   # chấm config.yaml hiện tại
    python -m tools.hieu_chinh --set "people_tracker.max_misses=[2, 3, 5]"
    python -m tools.hieu_chinh --set "detection.min_area=[200, 400]" --set "reid.match_threshold=[0.2, 0.4]"
    python -m tools.hieu_chinh --chi-dap-an                      # chỉ chạy cảnh có đáp án (nhanh hơn)

- Mỗi `--set` được thử RIÊNG: chỉ đổi đúng tham số đó, các tham số khác giữ như config.yaml.
  Luôn có thêm một lượt "hien_tai" (config.yaml không đổi) để so.
- Cảnh CÓ đáp án (camera có gt_tracking.csv, do nhóm khoanh bằng tools/annotate.py): in MOTA, IDF1 trung bình
  các camera sau khi nối tracklet, và số người nhận dạng lại đúng (chỉ ngoại hình | ngoại hình + sàn).
  ĐIỂM = (MOTA + IDF1) / 2, càng cao càng tốt.
- Cảnh CHƯA có đáp án: in bài chấm không cần đáp án (QĐ-34, src/evaluation/khong_dap_an.py) cho từng camera:
  số ID (ít là tốt), phủ người (cao là tốt), track ma (thấp là tốt), cú nhảy (thấp là tốt), track mỗi frame.
Kết quả ghi vào outputs/hieu_chinh/<thời gian>/bang_diem.csv, kèm config của từng lượt.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.common.utils import camera_video
from tools.cham_khong_dap_an import score_scene

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "data/canh"


def parse_set(text: str) -> tuple[str, list]:
    """'a.b=[1, 2]' → ('a.b', [1, 2]). Vế phải đọc như YAML; một giá trị đơn cũng được."""
    key, _, raw = text.partition("=")
    values = yaml.safe_load(raw)
    return key.strip(), values if isinstance(values, list) else [values]


def with_value(config: dict, dotted: str, value) -> dict:
    """Bản sao config với khóa dạng 'section.key' (hoặc 'section.sub.key') đổi thành value."""
    result = copy.deepcopy(config)
    node = result
    *parents, last = dotted.split(".")
    for part in parents:
        node = node[part]
    if last not in node:
        raise SystemExit(f"Không có khóa {dotted} trong config.yaml")
    node[last] = value
    return result


def make_variants(config: dict, sets: list[str]) -> list[tuple[str, dict]]:
    """Lượt 'hien_tai' cộng một lượt cho mỗi giá trị của mỗi --set."""
    variants = [("hien_tai", config)]
    for text in sets:
        key, values = parse_set(text)
        variants += [(f"{key}={value}", with_value(config, key, value)) for value in values]
    return variants


def run_scene(config_path: Path, scene: Path, out_dir: Path, threads: int) -> None:
    """Chạy run.py cho một cảnh; lỗi thì ghi log để xem lại.

    Mỗi lượt chỉ được dùng `threads` luồng CPU (OpenCV, NumPy), để nhiều lượt chạy song song
    không tranh nhau làm máy quá tải.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENCV_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               PYTHONUTF8="1")  # Windows: in tiếng Việt ra file log không bị lỗi mã hóa
    with (out_dir / "log.txt").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(ROOT / "run.py"), "--scene", str(scene), "--out", str(out_dir),
                        "--config", str(config_path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)


def scenes() -> list[Path]:
    """Mọi thư mục cảnh trong data/canh có video ở cam_a (cam_b có thể thiếu)."""
    return sorted(path for path in SCENE_ROOT.iterdir() if camera_video(path / "cam_a") is not None) if SCENE_ROOT.is_dir() else []


def has_answers(scene: Path) -> bool:
    """Cảnh có đáp án: camera A có gt_tracking.csv."""
    return (scene / "cam_a" / "gt_tracking.csv").exists()


def scored_metrics(run_dir: Path, scenes_with_answers: list[Path]) -> dict:
    """MOTA, IDF1 trung bình các camera (sau khi nối) và số người nhận dạng lại đúng, trên cảnh có đáp án."""
    mota, idf1, reid, reid_floor, people = [], [], 0, 0, 0
    for scene in scenes_with_answers:
        summary = json.loads((run_dir / scene.name / "summary.json").read_text(encoding="utf-8"))
        for camera in ("cam_a", "cam_b"):
            if camera in summary:
                mota.append(summary[camera]["sau_khi_noi"]["MOTA"])
                idf1.append(summary[camera]["sau_khi_noi"]["IDF1"])
        if "reid_ngoai_hinh" in summary:
            reid += int(summary["reid_ngoai_hinh"].split("/")[0])
            reid_floor += int(summary["reid_ngoai_hinh_san"].split("/")[0])
            people += int(summary["reid_ngoai_hinh"].split("/")[1])
    if not mota:
        return {}
    mean_mota, mean_idf1 = sum(mota) / len(mota), sum(idf1) / len(idf1)
    return {"MOTA": round(mean_mota, 3), "IDF1": round(mean_idf1, 3), "DIEM": round((mean_mota + mean_idf1) / 2, 3),
            "nhan_dang_lai": f"{reid}/{people} | {reid_floor}/{people}"}


def no_answer_scores(run_dir: Path, scene_list: list[Path]) -> dict:
    """Bài chấm không cần đáp án (QĐ-34) của từng camera: số ID, phủ người, track ma, cú nhảy."""
    result = {}
    for scene in scene_list:
        for camera, values in score_scene(run_dir / scene.name, scene).items():
            result[f"{scene.name}/{camera}"] = (f"{values['so_id']} ID, phủ {values['phu_nguoi']:.2f}, ma {values['track_ma']:.2f}, "
                                                f"nhảy {values['nhay']}, {values['track_moi_frame']} track/frame")
    return result


def main() -> None:
    """Chạy mọi lượt song song, in và ghi bảng điểm."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], help='ví dụ "people_tracker.max_misses=[5, 10, 20]"')
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--chi-dap-an", action="store_true", help="chỉ chạy các cảnh có đáp án (nhanh hơn)")
    parser.add_argument("--jobs", type=int, default=8, help="số lượt run.py chạy cùng lúc")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hieu_chinh" / time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_scenes = [scene for scene in scenes() if has_answers(scene) or not args.chi_dap_an]
    answered = [scene for scene in all_scenes if has_answers(scene)]
    if not all_scenes:
        raise SystemExit(f"Không có cảnh nào trong {SCENE_ROOT} (xem data/README.md)")
    variants = make_variants(config, args.set)
    jobs = []
    for index, (name, variant_config) in enumerate(variants):
        run_dir = args.out / f"{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ten.txt").write_text(name, encoding="utf-8")
        (run_dir / "config.yaml").write_text(yaml.safe_dump(variant_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        jobs += [(run_dir / "config.yaml", scene, run_dir / scene.name) for scene in all_scenes]
    print(f"{len(variants)} lượt × {len(all_scenes)} cảnh ({len(answered)} cảnh có đáp án), kết quả ở {args.out}", flush=True)
    threads = max(1, (os.cpu_count() or 1) // args.jobs)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda job: run_scene(*job, threads), jobs))
    table = []
    for index, (name, _) in enumerate(variants):
        run_dir = args.out / f"{index:02d}"
        row = {"luot": name, **scored_metrics(run_dir, answered)}
        row.update(no_answer_scores(run_dir, all_scenes))
        table.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    with (args.out / "bang_diem.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
