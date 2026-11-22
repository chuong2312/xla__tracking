"""Test tạo thư mục cảnh: chép video .mp4, ghi lại video định dạng khác thành .mp4."""
import cv2
import numpy as np

from tools.tao_canh import place


def _write_video(path, fourcc, frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10.0, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()


def test_mp4_is_copied(tmp_path):
    source = tmp_path / "a.mp4"
    _write_video(source, "mp4v")
    place(source, tmp_path / "scene" / "cam_a")
    target = tmp_path / "scene" / "cam_a" / "video.mp4"
    assert target.read_bytes() == source.read_bytes()


def test_avi_is_reencoded_to_mp4(tmp_path):
    source = tmp_path / "b.avi"
    _write_video(source, "MJPG", frames=7)
    place(source, tmp_path / "scene" / "cam_b")
    capture = cv2.VideoCapture(str(tmp_path / "scene" / "cam_b" / "video.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
