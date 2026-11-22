"""Hiệu chỉnh tham số: chạy toàn hệ thống với nhiều giá trị tham số, in bảng điểm để so sánh.

Cách dùng:
    python -m tools.hieu_chinh                                   # chấm config.yaml hiện tại
    python -m tools.hieu_chinh --set "people_tracker.max_misses=[5, 10, 20]"
    python -m tools.hieu_chinh --set "detection.mog2_learning_rate=[0.002, 0.0005]" --set "reid.match_threshold=[0.2, 0.4]"
    python -m tools.hieu_chinh --no-real                         # bỏ video thật cho nhanh (~1 phút thay vì ~3 phút)

- Mỗi `--set` được thử RIÊNG: chỉ đổi đúng tham số đó, các tham số khác giữ như config.yaml.
  Luôn có thêm một lượt "hien_tai" (config.yaml không đổi) để so.
- Video tổng hợp (data/synthetic_v2/s1…s5) có đáp án: in MOTA, IDF1 trung bình 10 camera sau khi
  nối tracklet, và số người nhận dạng lại đúng (chỉ ngoại hình | ngoại hình + sàn).
  ĐIỂM = (MOTA + IDF1) / 2, càng cao càng tốt.
- Video thật (data/real/epfl_4p_c0_c1) chưa có đáp án: chỉ in số ID và số track mỗi frame của
  từng camera. Cảnh có tối đa 4 người, nên số ID càng ít càng tốt, nhưng track mỗi frame không
  nên tụt (tụt là đang bỏ sót người).
Kết quả ghi vào outputs/hieu_chinh/<thời gian>/bang_diem.csv, kèm config của từng lượt.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = [ROOT / "data/synthetic_v2" / f"s{i}" for i in range(1, 6)]
REAL = [ROOT / "data/real/epfl_4p_c0_c1"]
ACTIVE = ("confirmed", "merged", "static")


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


def run_scene(config_path: Path, scene: Path, out_dir: Path) -> None:
    """Chạy run.py cho một cảnh; lỗi thì ghi log để xem lại."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "log.txt").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(ROOT / "run.py"), "--scene", str(scene), "--out", str(out_dir),
                        "--config", str(config_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)


def synthetic_scores(run_dir: Path, scenes: list[Path]) -> dict:
    """MOTA, IDF1 trung bình (sau khi nối) và số người nhận dạng lại đúng trên các cảnh có đáp án."""
    mota, idf1, reid, reid_floor, people = [], [], 0, 0, 0
    for scene in scenes:
        summary = json.loads((run_dir / scene.name / "summary.json").read_text(encoding="utf-8"))
        for camera in ("cam_a", "cam_b"):
            mota.append(summary[camera]["sau_khi_noi"]["MOTA"])
            idf1.append(summary[camera]["sau_khi_noi"]["IDF1"])
        reid += int(summary["reid_ngoai_hinh"].split("/")[0])
        reid_floor += int(summary["reid_ngoai_hinh_san"].split("/")[0])
        people += int(summary["reid_ngoai_hinh"].split("/")[1])
    mean_mota, mean_idf1 = sum(mota) / len(mota), sum(idf1) / len(idf1)
    return {"MOTA": round(mean_mota, 3), "IDF1": round(mean_idf1, 3), "DIEM": round((mean_mota + mean_idf1) / 2, 3),
            "nhan_dang_lai": f"{reid}/{people} | {reid_floor}/{people}"}


def real_counts(run_dir: Path, scenes: list[Path]) -> dict:
    """Số ID và số track mỗi frame (sau khi nối) của từng camera trên video thật."""
    result = {}
    for scene in scenes:
        for camera in ("cam_a", "cam_b"):
            with (run_dir / scene.name / camera / "tracks_linked.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            frames = len({r["frame"] for r in rows}) or 1
            active = [r for r in rows if r["state"] in ACTIVE]
            ids = len(collections.Counter(r["track_id"] for r in active))
            result[f"{scene.name}/{camera}"] = f"{ids} ID, {len(active) / frames:.2f} track/frame"
    return result


def main() -> None:
    """Chạy mọi lượt song song, in và ghi bảng điểm."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], help='ví dụ "people_tracker.max_misses=[5, 10, 20]"')
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--no-real", action="store_true", help="bỏ video thật (nhanh hơn)")
    parser.add_argument("--jobs", type=int, default=8, help="số lượt run.py chạy cùng lúc")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hieu_chinh" / time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    synthetic = [s for s in SYNTHETIC if s.exists()]
    real = [] if args.no_real else [s for s in REAL if s.exists()]
    variants = make_variants(config, args.set)
    jobs = []
    for index, (name, variant_config) in enumerate(variants):
        run_dir = args.out / f"{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ten.txt").write_text(name, encoding="utf-8")
        (run_dir / "config.yaml").write_text(yaml.safe_dump(variant_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        jobs += [(run_dir / "config.yaml", scene, run_dir / scene.name) for scene in real + synthetic]
    print(f"{len(variants)} lượt × {len(real) + len(synthetic)} cảnh, kết quả ở {args.out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda job: run_scene(*job), jobs))
    table = []
    for index, (name, _) in enumerate(variants):
        run_dir = args.out / f"{index:02d}"
        row = {"luot": name, **synthetic_scores(run_dir, synthetic)} if synthetic else {"luot": name}
        row.update(real_counts(run_dir, real))
        table.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    with (args.out / "bang_diem.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
