"""Dữ liệu của video tổng hợp v2: bố trí hành lang và 5 kịch bản (QĐ-22).

Bố trí "hanh_lang": hành lang rộng 3 m, dài 10 m. Trục x đi ngang hành lang (0 là tường trái,
3 là tường phải), trục y đi dọc hành lang (0 là đầu xa, 10 là đầu gần camera).
Hai camera đặt ở CÙNG một đầu hành lang (đầu gần): cam_a ở góc trái, cam_b ở góc phải,
cả hai nhìn dọc vào trong, nên cùng thấy gần hết hành lang từ hai góc lệch nhau.
Người đi về phía camera thì to dần, đi ra xa thì nhỏ dần.
"""
from __future__ import annotations

from tools.synthetic_scene import Layout, Person, Scenario

# Ngoại hình: (màu áo BGR, màu quần BGR, hoa văn).
LOOKS = {
    1: ((40, 40, 200), (120, 60, 20), "soc_ngang"),
    2: ((60, 170, 60), (90, 90, 90), "soc_doc"),
    3: ((40, 200, 220), (30, 30, 30), "o_vuong"),
    4: ((160, 60, 140), (40, 70, 120), "cham"),
}


def make_person(gt_id: int, waypoints, look_id: int | None = None) -> Person:
    """Tạo một người; `look_id` cho phép hai người mặc giống hệt nhau (kịch bản s4)."""
    top, bottom, pattern = LOOKS[look_id or gt_id]
    return Person(gt_id, top, bottom, pattern, tuple(waypoints))


_FLOOR = ((0, 0), (3, 0), (3, 10), (0, 10))

LAYOUTS = {
    "hanh_lang": Layout(
        name="hanh_lang",
        floor_rect_m=(0.0, 0.0, 3.0, 10.0),
        camera_views={
            "cam_a": (_FLOOR, ((190, 70), (430, 82), (700, 370), (60, 345))),
            "cam_b": (_FLOOR, ((210, 82), (450, 70), (580, 345), (-60, 370))),
        },
        tape_marks_m=((0.5, 3.0), (2.5, 3.0), (0.5, 7.0), (2.5, 7.0), (1.5, 5.0), (1.0, 8.5)),
    ),
}

_CORRIDOR = LAYOUTS["hanh_lang"]

SCENARIOS = {
    "s1": Scenario("s1", _CORRIDOR, (
        make_person(1, [(2.0, 0.6, 9.8), (10.0, 0.6, 0.8)]),
        make_person(2, [(2.5, 2.4, 0.5), (10.5, 2.4, 9.8)]),
        make_person(3, [(4.0, 1.2, 9.8), (11.5, 1.2, 1.5)]),
        make_person(4, [(3.5, 1.8, 0.5), (11.5, 1.8, 9.0)]),
    ), "đi thưa theo hai chiều, mỗi người một làn"),
    "s2": Scenario("s2", _CORRIDOR, (
        make_person(1, [(2.0, 0.6, 9.8), (6.0, 1.5, 5.0), (10.0, 2.4, 0.8)]),
        make_person(2, [(2.0, 2.4, 0.5), (6.0, 1.5, 5.2), (10.0, 0.6, 9.8)]),
        make_person(3, [(3.0, 1.5, 9.8), (11.0, 1.5, 1.0)]),
        make_person(4, [(4.0, 0.9, 0.5), (11.5, 2.1, 9.5)]),
    ), "đi ngược chiều, đổi làn, cắt nhau và che nhau"),
    "s3": Scenario("s3", _CORRIDOR, (
        make_person(1, [(2.0, 1.0, 0.8), (4.5, 1.0, 5.0), (9.5, 1.0, 5.0), (11.5, 1.0, 7.5)]),
        make_person(2, [(3.0, 1.4, 9.8), (10.0, 1.4, 1.0)]),
        make_person(3, [(3.5, 2.4, 0.5), (11.0, 2.4, 9.5)]),
        make_person(4, [(5.0, 0.5, 9.8), (12.0, 0.5, 2.0)]),
    ), "người 1 đứng yên 5 giây, người 2 đi ngang trước mặt"),
    "s4": Scenario("s4", _CORRIDOR, (
        make_person(1, [(2.0, 0.8, 9.8), (10.0, 2.2, 0.8)]),
        make_person(2, [(2.0, 2.2, 0.8), (10.0, 0.8, 9.8)], look_id=1),
        make_person(3, [(3.5, 1.5, 9.8), (11.0, 1.5, 1.5)]),
        make_person(4, [(4.0, 2.6, 0.5), (11.5, 2.6, 9.0)]),
    ), "người 1 và người 2 mặc giống hệt nhau, đi ngược chiều cắt nhau"),
    "s5": Scenario("s5", _CORRIDOR, (
        make_person(1, [(2.0, 1.2, 0.5), (6.5, 1.2, 5.5), (10.5, 0.5, 9.8)]),
        make_person(2, [(2.0, 1.6, 0.5), (6.5, 1.6, 5.5), (10.5, 2.5, 9.8)]),
        make_person(3, [(3.0, 0.5, 9.8), (10.5, 0.5, 1.0)]),
        make_person(4, [(4.0, 2.5, 9.8), (11.5, 2.5, 1.5)]),
    ), "người 1 và người 2 đi sát nhau thành nhóm rồi tách ra"),
}
