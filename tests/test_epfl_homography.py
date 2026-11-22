"""Test đọc file hiệu chỉnh EPFL và tính homography sàn A → B."""
import numpy as np

from src.homography import project
from tools.epfl_homography import floor_to_floor, read_ground_homographies

CALIB = """# Camera 0
#
# Ground plane homography
2 0 10
0 2 20
0 0 1

# Head plane homography
1 0 0
0 1 0
0 0 1

# Camera 1
# Ground plane homography
1 0 -5
0 1 5
0 0 1
"""


def test_reads_only_ground_plane_of_each_camera(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(CALIB, encoding="utf-8")
    result = read_ground_homographies(path)
    assert sorted(result) == [0, 1]
    assert result[0][0, 2] == 10 and result[1][1, 2] == 5


def test_point_maps_to_same_floor_position_in_both_cameras():
    g_a = np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]], dtype=float)
    g_b = np.array([[1, 0, -5], [0, 1, 5], [0, 0, 1]], dtype=float)
    h_ab = floor_to_floor(g_a, g_b)
    point_a = np.array([[7.0, 3.0]])
    point_b = project(h_ab, point_a)
    assert np.allclose(project(g_a, point_a), project(g_b, point_b))
