"""Dữ liệu của video tổng hợp v2: các bố trí sàn – camera và các kịch bản.

Có hai bố trí:
- "san_chung" (QĐ-02): sàn 6 m × 4 m; hai camera cùng nhìn cả khu vực từ hai góc.
- "hanh_lang" (QĐ-17): hành lang dài 8 m, rộng 2,5 m, có cửa ở tường trái.
  Camera A đặt ở đầu hành lang, nhìn dọc hành lang. Camera B gắn ở tường đối diện
  cửa, chỉ thấy đoạn hành lang quanh cửa. Hai camera chỉ chung nhau vùng quanh cửa.
Trục x của sàn chạy dọc hành lang hoặc từ trái sang phải; trục y đi từ xa tới gần
camera A (bố trí san_chung), hoặc từ tường có cửa sang tường đối diện (bố trí hanh_lang).
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


LAYOUTS = {
    "san_chung": Layout(
        name="san_chung",
        floor_rect_m=(0.0, 0.0, 6.0, 4.0),
        room_rect_m=None,
        camera_views={
            "cam_a": (((0, 0), (6, 0), (6, 4), (0, 4)), ((150, 120), (490, 120), (630, 350), (10, 350))),
            "cam_b": (((0, 0), (6, 0), (6, 4), (0, 4)), ((330, 105), (600, 190), (330, 355), (60, 200))),
        },
        door_m=((2.2, 1.0), (3.8, 1.0)),
        door_inside_point_m=(3.0, 2.5),
        tape_marks_m=((1.0, 0.8), (3.0, 0.5), (5.0, 0.8), (1.0, 3.2), (3.0, 3.5), (5.0, 3.2)),
    ),
    "hanh_lang": Layout(
        name="hanh_lang",
        floor_rect_m=(0.0, 0.0, 8.0, 2.5),
        room_rect_m=(4.5, -1.0, 5.5, 0.0),
        camera_views={
            "cam_a": (((0, 0), (8, 0), (8, 2.5), (0, 2.5)), ((20, 350), (210, 115), (430, 115), (620, 350))),
            "cam_b": (((3, 0), (7, 0), (7, 2.5), (3, 2.5)), ((150, 110), (490, 110), (690, 360), (-50, 360))),
        },
        door_m=((4.5, 0.0), (5.5, 0.0)),
        door_inside_point_m=(5.0, -1.0),
        tape_marks_m=((3.5, 0.4), (6.5, 0.4), (3.5, 2.1), (6.5, 2.1), (5.0, 1.25), (4.2, 0.9)),
    ),
}

_SHARED = LAYOUTS["san_chung"]
_CORRIDOR = LAYOUTS["hanh_lang"]

SCENARIOS = {
    "s1": Scenario("s1", _SHARED, (
        make_person(1, [(2.0, -0.5, 0.8), (9.0, 6.5, 0.8)]),
        make_person(2, [(3.0, -0.5, 1.8), (10.0, 6.5, 1.8)]),
        make_person(3, [(4.0, -0.5, 2.8), (11.0, 6.5, 2.8)]),
        make_person(4, [(5.0, -0.5, 3.6), (12.0, 6.5, 3.6)]),
    ), "4 người đi 4 làn riêng, lệch giờ nhau"),
    "s2": Scenario("s2", _SHARED, (
        make_person(1, [(2.0, -0.5, 0.5), (9.0, 6.5, 3.5)]),
        make_person(2, [(2.0, 6.5, 0.5), (9.0, -0.5, 3.5)]),
        make_person(3, [(3.0, -0.5, 2.0), (10.0, 6.5, 2.0)]),
        make_person(4, [(4.0, 3.0, -0.5), (11.0, 3.0, 4.5)]),
    ), "đi chéo, cắt ngang nhau"),
    "s3": Scenario("s3", _SHARED, (
        make_person(1, [(2.0, -0.5, 2.0), (5.0, 3.0, 2.0), (10.0, 3.0, 2.0), (12.0, 5.0, 2.0)]),
        make_person(2, [(2.5, 6.5, 1.0), (10.0, -0.5, 1.0)]),
        make_person(3, [(4.0, -0.5, 3.2), (11.0, 6.5, 3.2)]),
        make_person(4, [(3.0, 6.5, 0.4), (11.0, -0.5, 0.4)]),
    ), "người 1 đứng yên 5 giây"),
    "s4": Scenario("s4", _SHARED, (
        make_person(1, [(2.0, -0.5, 1.5), (9.0, 6.5, 2.5)]),
        make_person(2, [(2.0, 6.5, 1.5), (9.0, -0.5, 2.5)], look_id=1),
        make_person(3, [(3.0, -0.5, 0.6), (11.0, 6.5, 0.6)]),
        make_person(4, [(4.0, 6.5, 3.4), (12.0, -0.5, 3.4)]),
    ), "người 1 và người 2 mặc giống hệt nhau, đi cắt nhau"),
    "door": Scenario("door", _SHARED, (
        make_person(1, [(2.0, 3.0, -0.5), (6.0, 3.0, 3.0), (8.0, 6.5, 3.0)]),
        make_person(2, [(3.0, 2.6, 3.5), (7.0, 2.6, -0.5)]),
        make_person(3, [(6.0, 3.4, -0.5), (9.5, 3.4, 3.0)]),
        make_person(4, [(5.5, 2.4, 3.0), (9.0, 2.4, -0.5)]),
    ), "2 vào, 2 ra; người 3 và 4 qua cửa cùng lúc, ngược chiều"),
    "hanh_lang_cua": Scenario("hanh_lang_cua", _CORRIDOR, (
        make_person(1, [(2.0, 0.3, 1.2), (6.0, 4.6, 1.2), (7.0, 5.0, 0.3), (7.6, 5.0, -0.4)]),
        make_person(2, [(2.5, 0.3, 1.9), (10.0, 8.0, 1.9)]),
        make_person(3, [(5.0, 5.0, -0.4), (5.6, 5.0, 0.5), (9.5, 0.3, 0.9)]),
        make_person(4, [(2.6, 0.3, 0.8), (8.0, 4.8, 0.8), (8.8, 5.2, -0.4)]),
    ), "người 1 rẽ vào cửa; người 4 vào sau 1,3 giây; người 3 từ phòng đi ra; người 2 đi ngang qua cửa không vào"),
}
