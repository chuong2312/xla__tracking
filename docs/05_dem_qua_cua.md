# 05. Đếm người vào/ra qua cửa

> Code: `src/door.py`. Ảnh minh họa: `python -m tools.explain_door`.
> Phần này **bắt buộc** theo yêu cầu của thầy (QĐ-12). Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Đếm xem có bao nhiêu lượt người **đi vào** và bao nhiêu lượt **đi ra** qua cửa, kể cả khi 2 người qua cùng lúc theo hai chiều ngược nhau.

## Cách làm

1. **Điểm chân:** mỗi người được đại diện bằng điểm giữa cạnh dưới của box, tức chỗ chân chạm sàn. Dùng điểm chân vì cửa nằm trên sàn; dùng tâm box thì người cao, người thấp sẽ cho kết quả khác nhau.
2. **Vạch cửa:** một đoạn thẳng vẽ ngang lối cửa trên ảnh.
3. **Xác định phía:** tính khoảng cách có dấu từ điểm chân tới vạch. Dương là một bên, âm là bên kia.
4. **Vùng đệm:** hai bên vạch có một dải hẹp. Khi chân nằm trong dải này, chương trình **chưa quyết định** người ở phía nào. Nhờ vậy người đứng rung trên vạch không bị đếm nhiều lần.
5. **Phát sự kiện:** khi phía đã xác định của một người đổi dấu, và chân nằm trong bề rộng cửa, thì ghi một lượt: sang phía "bên trong" là **VÀO**, sang phía còn lại là **RA**.
6. **Chỉ dùng vị trí, không dùng màu.** Test hồi quy bắt buộc "IN 1 / OUT 1" chỉ đưa vào điểm chân.

## Công thức

Vạch cửa đi từ P1 = (x1, y1) tới P2 = (x2, y2). Với điểm chân p = (px, py):

- **Khoảng cách có dấu:** `d = ((x2 − x1)·(py − y1) − (y2 − y1)·(px − x1)) / ‖P2 − P1‖`. Trên ảnh (trục y hướng xuống), d > 0 nghĩa là p nằm **bên phải** khi đi từ P1 sang P2.
- **Có nằm trong bề rộng cửa không:** `t = ((p − P1) · (P2 − P1)) / ‖P2 − P1‖²`, cần 0 ≤ t ≤ 1. Người đi vòng qua đầu vạch, ngoài bề rộng cửa, thì không tính.
- **Vùng đệm:** `|d| ≤ crossing_margin` thì chưa quyết định phía.

## Tham số (`config.yaml`)

| Tham số | Ý nghĩa |
|---|---|
| `events.crossing_margin` (5) | Độ rộng vùng đệm mỗi bên vạch (pixel) |
| `events.inside_side` (`positive`) | Phía nào là "bên trong": `positive` là bên phải khi đi từ đầu tới cuối vạch, `negative` là bên trái |
| `scene.door_line` | Tọa độ hai đầu vạch cửa trên ảnh |

## Ảnh minh họa

`python -m tools.explain_door` sinh ảnh `outputs/explain/dem_qua_cua/door_cam_a.png`. Ảnh vẽ vạch cửa, ranh giới vùng đệm, đường đi của điểm chân 4 người, và các chấm đánh số ở chỗ phát sự kiện (xanh là vào, đỏ là ra). Trên kịch bản `door`, cả hai camera đều đếm đúng **VÀO 2, RA 2**, gồm cả cặp người 3 và người 4 qua cửa cùng lúc (frame 229).

## Khi nào hỏng

- **Chân bị che hoặc box lệch:** điểm chân sai nên phía bị xác định sai.
- **Track bị đứt ngay ở cửa:** người mới hiện ra ở phía bên kia nhận ID mới, nên bộ đếm không biết người đó vừa đi qua. Vì vậy tracker phải gọi `update` **trước khi** kết thúc track (XLA 2.17).
- **Vùng đệm quá rộng:** người chỉ bước qua vạch một chút sẽ không được đếm. **Quá hẹp:** người đứng rung trên vạch dễ bị đếm thừa.
- **Hai người dính thành một box khi qua cửa:** chỉ đếm được một lượt.

## Thầy có thể hỏi

- *"Vì sao dùng điểm chân mà không dùng tâm box?"* Cửa nằm trên sàn. Tâm box của người cao và người thấp ở độ cao khác nhau, còn chân thì luôn chạm sàn.
- *"Vùng đệm để làm gì?"* Để người đứng rung đúng trên vạch không bị đếm vào, ra, vào, ra liên tục.
- *"Có dùng màu không?"* Không. Chỉ dùng vị trí.
