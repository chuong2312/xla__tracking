"""Test tạo thư mục cảnh từ video thật của nhóm: chép nguyên file, giữ đuôi gốc, không mã hóa lại."""
import pytest

from du_lieu_that import LAB4P_VIDEOS
from src.common.utils import camera_video
from tools.tao_canh import place


def _need_videos():
    if not all(video.exists() for video in LAB4P_VIDEOS[:2]):
        pytest.skip("Thiếu video thật trong data/goc/epfl_lab4p (xem data/README.md)")


def test_video_is_copied_with_its_own_extension(tmp_path):
    _need_videos()
    source = LAB4P_VIDEOS[0]
    target = place(source, tmp_path / "canh" / "cam_a")
    assert target.name == "video.avi"
    assert camera_video(tmp_path / "canh" / "cam_a") == target
    assert target.read_bytes() == source.read_bytes()


def test_new_video_replaces_the_old_one(tmp_path):
    _need_videos()
    place(LAB4P_VIDEOS[0], tmp_path / "cam_a")
    place(LAB4P_VIDEOS[1], tmp_path / "cam_a")
    files = list((tmp_path / "cam_a").iterdir())
    assert [path.name for path in files] == ["video.avi"]
    assert files[0].read_bytes() == LAB4P_VIDEOS[1].read_bytes()
