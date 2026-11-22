"""Đọc và kiểm tra cấu hình YAML của đồ án."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class SceneConfig:
    resize_width: int
    resize_height: int
    roi: list[int] | None
    door_line: list[int]

@dataclass(frozen=True)
class AppConfig:
    scene: SceneConfig
    detector: dict[str, Any]
    tracker: dict[str, Any]
    events: dict[str, Any]
    features: dict[str, Any]
    reid: dict[str, Any]
    benchmark: dict[str, Any]

_SECTIONS = ("scene", "detector", "tracker", "events", "features", "reid", "benchmark")
def load_config(path: str | Path) -> AppConfig:
    """Nạp cấu hình và báo lỗi có ngữ cảnh nếu schema không hợp lệ."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Không đọc được config '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config phải là mapping YAML")
    missing = [key for key in _SECTIONS if key not in data]
    if missing:
        raise ValueError(f"Thiếu section: {', '.join(missing)}")
    for key in _SECTIONS[1:]:
        if not isinstance(data[key], dict):
            raise ValueError(f"Section '{key}' phải là mapping")
    scene = data["scene"]
    if not isinstance(scene, dict):
        raise ValueError("Section 'scene' phải là mapping")
    for key in ("resize_width", "resize_height", "roi", "door_line"):
        if key not in scene:
            raise ValueError(f"Thiếu key 'scene.{key}'")
    for key in ("resize_width", "resize_height"):
        if type(scene[key]) is not int or scene[key] <= 0:
            raise ValueError(f"'scene.{key}' phải là số nguyên dương")
    roi, line = scene["roi"], scene["door_line"]
    if roi is not None and (not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi)):
        raise ValueError("'scene.roi' phải là null hoặc danh sách 4 số nguyên")
    if not isinstance(line, list) or len(line) != 4 or any(type(v) is not int for v in line):
        raise ValueError("'scene.door_line' phải là danh sách 4 số nguyên")
    return AppConfig(SceneConfig(scene["resize_width"], scene["resize_height"], roi, line), **{k: data[k] for k in _SECTIONS[1:]})
