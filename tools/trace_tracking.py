"""Chạy pipeline với trace JSONL và manifest provenance cho từng run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

import cv2
import numpy as np
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: str | Path) -> str:
    """Tính SHA-256 trên đúng bytes của một file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_status(root: Path, path: Path) -> str:
    """Đọc trạng thái Git của một file mà không thay đổi index."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return "external_input"
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", str(relative)],
        cwd=root, check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() or "clean"


def collect_provenance_files(
    repo_root: str | Path,
    config_path: str | Path,
    video_path: str | Path,
    gt_path: str | Path,
) -> list[dict[str, Any]]:
    """Hash source, config, tests và toàn bộ synthetic input cùng scenario."""
    root = Path(repo_root).resolve()
    config = Path(config_path).resolve()
    video = Path(video_path).resolve()
    gt = Path(gt_path).resolve()
    roles: dict[Path, set[str]] = {}

    def add(path: Path, role: str) -> None:
        resolved = path.resolve()
        if resolved.is_file():
            roles.setdefault(resolved, set()).add(role)

    add(config, "config")
    add(root / "legacy/run_tracking.py", "source")
    add(root / "tools" / "trace_tracking.py", "source")
    add(root / "tools" / "make_synthetic.py", "synthetic_generator")
    for directory, role in ((root / "src", "source"), (root / "legacy", "source"), (root / "tests", "test")):
        if directory.exists():
            for path in directory.rglob("*.py"):
                add(path, role)

    add(video, "input_video")
    add(gt, "ground_truth")
    try:
        common = Path(os.path.commonpath([video.parent, gt.parent]))
        common.relative_to((root / "data" / "synthetic").resolve())
        if common.is_dir():
            for path in common.rglob("*"):
                if path.is_file():
                    add(path, "synthetic_input")
    except ValueError:
        pass
    metadata = video.parent / "metadata.json"
    add(metadata, "input_metadata")

    records = []
    for path in sorted(roles, key=lambda value: str(value)):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = str(path)
        records.append({
            "relative_path": relative,
            "roles": sorted(roles[path]),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_status": _git_status(root, path),
        })
    return records


def _git_value(root: Path, *args: str) -> str | None:
    """Trả kết quả Git read-only hoặc None nếu lệnh không khả dụng."""
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_run_manifest(
    repo_root: str | Path,
    config_path: str | Path,
    video_path: str | Path,
    gt_path: str | Path,
) -> dict[str, Any]:
    """Tạo manifest có hash file và trạng thái working tree của một run."""
    root = Path(repo_root).resolve()
    config = Path(config_path).resolve()
    video = Path(video_path).resolve()
    gt = Path(gt_path).resolve()
    config_snapshot = config.read_text(encoding="utf-8")
    records = collect_provenance_files(root, config, video, gt)
    head = _git_value(root, "rev-parse", "HEAD")
    status = _git_value(root, "status", "--porcelain")
    diff = _git_value(root, "diff", "--binary") or ""
    staged_diff = _git_value(root, "diff", "--cached", "--binary") or ""
    metadata_path = video.parent / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    try:
        video_name = video.relative_to(root).as_posix()
    except ValueError:
        video_name = str(video)
    try:
        gt_name = gt.relative_to(root).as_posix()
    except ValueError:
        gt_name = str(gt)
    return {
        "event": "run_manifest",
        "run_id": str(uuid.uuid4()),
        "git_head": head,
        "git_status_porcelain": status,
        "working_tree_diff_sha256": hashlib.sha256((diff + staged_diff).encode("utf-8")).hexdigest(),
        "config_snapshot": config_snapshot,
        "config_sha256": hashlib.sha256(config_snapshot.encode("utf-8")).hexdigest(),
        "input_provenance": {
            "video": video_name,
            "ground_truth": gt_name,
            "scenario": metadata.get("scenario"),
            "synthetic": metadata.get("synthetic"),
            "random_seed": metadata.get("seed"),
            "seed_scope": "synthetic_generator" if metadata.get("seed") is not None else None,
        },
        "software": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "files": records,
    }


class _RunJsonHandler(logging.StreamHandler):
    """Gắn run_id vào từng sự kiện tracker và ghi JSONL."""
    def __init__(self, stream, run_id: str):
        super().__init__(stream)
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
            payload["run_id"] = self.run_id
            self.stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            self.flush()
        except (OSError, TypeError, ValueError):
            self.handleError(record)


def main() -> None:
    """Chạy detector/tracker; tạo track CSV và diagnostics mà không ghi đè."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--tracks-out", required=True)
    parser.add_argument("--diagnostics-out", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    from legacy.config import load_config
    from legacy.detector import Detector
    from legacy.tracker import Tracker
    from src.common.utils import iter_video

    config_path = Path(args.config).resolve()
    video_path = Path(args.video).resolve()
    gt_path = Path(args.gt).resolve()
    tracks_path = Path(args.tracks_out).resolve()
    diagnostics_path = Path(args.diagnostics_out).resolve()
    if tracks_path == diagnostics_path:
        parser.error("--tracks-out và --diagnostics-out phải khác nhau")
    for path in (tracks_path, diagnostics_path):
        if path.exists():
            parser.error(f"Không ghi đè file đã tồn tại: {path}")
    cfg = load_config(config_path)
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = build_run_manifest(ROOT, config_path, video_path, gt_path)
    detector = Detector({"detector": cfg.detector, "scene": raw_cfg["scene"]})
    tracker = Tracker(cfg)
    size = (cfg.scene.resize_width, cfg.scene.resize_height)
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("legacy.tracker")
    previous_level, previous_propagate = logger.level, logger.propagate
    handler = None
    try:
        with tracks_path.open("x", newline="", encoding="utf-8") as tracks_stream, diagnostics_path.open("x", encoding="utf-8") as diagnostics_stream:
            diagnostics_stream.write(json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n")
            handler = _RunJsonHandler(diagnostics_stream, manifest["run_id"])
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            writer = csv.writer(tracks_stream)
            writer.writerow(["frame", "track_id", "state", "x", "y", "w", "h", "n_points"])
            frame_count = 0
            for frame_index, frame in iter_video(video_path, size):
                detections = detector.detect(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tracks = tracker.update(gray, detections, frame_bgr=frame)
                for track in tracks:
                    writer.writerow([frame_index, track.id, track.state, *track.box, len(track.points)])
                frame_count += 1
            diagnostics_stream.write(json.dumps({
                "event": "run_complete", "run_id": manifest["run_id"],
                "frame_count": frame_count,
            }, sort_keys=True) + "\n")
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    print(f"run_id={manifest['run_id']} frames={frame_count}")


if __name__ == "__main__":
    main()
