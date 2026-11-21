# 07. Ghép lượt qua cửa giữa hai camera theo thời gian

> Code: `src/door_matching.py`. Ảnh minh họa: `python -m tools.explain_door_matching`.
> Dùng cho bố trí "hành lang + cửa" (QĐ-17). Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Bố trí cảnh cửa:
- **Camera A** đặt ở đầu hành lang, thấy người đi dọc hành lang rồi rẽ vào cửa.
- **Camera B** gắn ở tường đối diện cửa, thấy người đi qua cửa.

Hai camera chỉ cùng thấy vùng quanh cửa. Câu hỏi: người mà camera A thấy rẽ vào cửa là người nào trong camera B?

## Ý tưởng

**Hai camera đã đồng bộ giờ** nhờ cái vỗ tay lúc bắt đầu quay. Nếu camera A thấy *một người VÀO* lúc giây 7,3, và camera B cũng thấy *một người VÀO* lúc giây 7,0, thì gần như chắc đó là **cùng một người**.

- Cách này **không cần nhìn quần áo**, nên hai người mặc giống nhau vẫn phân biệt được, miễn là họ không qua cửa cùng lúc.
- Mỗi camera tự đếm lượt qua cửa bằng bộ đếm ở [05_dem_qua_cua.md](05_dem_qua_cua.md). Module này chỉ **ghép các lượt** của hai camera với nhau.

## Cách làm

1. Lấy danh sách lượt qua cửa của từng camera. Mỗi lượt gồm: frame, số track, VÀO hay RA.
2. Với mỗi cặp (lượt ở A, lượt ở B), tính **độ lệch thời gian** = |frame A − frame B| / fps, đơn vị giây.
3. **Cấm ghép** khi:
   - khác chiều (một VÀO, một RA), hoặc
   - lệch quá `reid.door_event_max_gap_s` giây.
4. Ghép một-một bằng **Hungarian** ([03_ghep_cap_hungarian.md](03_ghep_cap_hungarian.md)) sao cho tổng độ lệch nhỏ nhất.
5. Mỗi cặp được ghép cho biết: "track số mấy ở camera A chính là track số mấy ở camera B".

## Công thức

- `lệch = |frame_A − frame_B| / fps` (giây).
- Cặp được phép khi `cùng chiều` và `lệch ≤ door_event_max_gap_s`.
- Hungarian chọn các cặp làm `Σ lệch` nhỏ nhất.

Số frame của hai camera phải theo **cùng một mốc giờ**. Nếu camera B bắt đầu quay muộn hơn thì phải cộng `frame_offset` vào trước khi ghép.

## Tham số (`config.yaml`)

| Tham số | Ý nghĩa |
|---|---|
| `reid.door_event_max_gap_s` (1,5) | Hai lượt cùng chiều lệch nhau quá bấy nhiêu giây thì không được ghép |

Hai lượt của cùng một người không trùng giờ tuyệt đối, vì camera A và camera B nhìn cửa từ hai góc khác nhau. Trên video tổng hợp, độ lệch thật là 0,03–0,37 giây. Ngoài đời còn có thể có người dừng lại ở cửa, nên đặt rộng hơn (1,5 giây), rồi hiệu chỉnh lại trên video dev.

## Ảnh minh họa

`python -m tools.explain_door_matching` sinh ảnh `outputs/explain/ghep_su_kien_cua/hanh_lang_cua.png`:
- **Hàng trên:** các lượt qua cửa ở camera A.
- **Hàng dưới:** các lượt qua cửa ở camera B. Số track được cố ý đánh khác (cộng 100), vì ngoài đời mỗi camera tự đặt số riêng.
- **Đường xanh dương:** các cặp được ghép.

Kết quả: track 1 ↔ 101, 3 ↔ 103, 4 ↔ 104, tức đúng cùng một người. Người 2 đi ngang qua cửa mà không vào nên không có lượt nào, và không bị ghép nhầm.

## Khi nào hỏng

- **Hai người qua cửa gần như cùng lúc, cùng chiều:** không biết ai là ai. Khi đó phải dựa thêm vào ngoại hình (SIFT, màu) và vị trí trên sàn (homography).
- **Một camera bỏ sót lượt qua cửa**, ví dụ người bị che đúng lúc qua cửa: lượt còn lại không có cặp, hoặc tệ hơn là bị ghép nhầm với lượt của người khác nếu người đó qua cửa gần giờ đó.
- **Hai video không đồng bộ giờ:** cả module mất tác dụng. Vì vậy vỗ tay lúc bắt đầu quay là **bắt buộc**.
- **Camera A nhìn cửa rất nghiêng** (cửa ở tường bên): vạch cửa trên ảnh của A rất ngắn, đếm kém tin cậy hơn camera B.

## Thầy có thể hỏi

- *"Làm sao biết người ở camera A là ai ở camera B?"* Với cảnh cửa: hai camera đã đồng bộ giờ, nên lượt qua cửa cùng chiều xảy ra gần như cùng lúc ở hai camera được coi là cùng một người. Ghép bằng Hungarian theo độ lệch thời gian.
- *"Hai người vào cùng lúc thì sao?"* Thời gian không đủ để phân biệt; phải dùng thêm ngoại hình và vị trí trên sàn. Đây là hạn chế đã biết và được ghi trong báo cáo.
