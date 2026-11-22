from pathlib import Path
import pytest
from legacy.config import load_config

def test_load_default_config():
    c=load_config(Path(__file__).parents[2]/"config.yaml"); assert c.scene.resize_width==640

def test_rejects_missing_section(tmp_path):
    p=tmp_path/"bad.yaml"; p.write_text("scene: {}", encoding="utf-8")
    with pytest.raises(ValueError,match="Thiếu section"): load_config(p)
