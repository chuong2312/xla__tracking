# 11. Tracker mới: bám người và giữ tên (bước D)

> Code: `src/tracking/people_tracker.py` (điều phối), `src/tracking/track.py` (hồ sơ một người), `src/tracking/optical_flow.py` (Lucas-Kanade),
> `src/tracking/blob_analysis.py` (cụm gộp / tách), `src/tracking/recovery.py` (ghép lại sau khi tách). Dùng lại `src/common/kalman.py` và `src/common/assignment.py`.
> Tham số ở section `people_tracker` của `config.yaml`. Tracker cũ (`legacy/tracker.py`) được giữ lại để so sánh.

## Bài toán

Nối các box của từng tấm ảnh thành **đường đi của từng người**, mỗi người giữ một tên (ID) từ lúc xuất hiện đến lúc đi khỏi, kể cả khi bị che, đứng yên hay đi sát người khác.

## Mỗi tấm ảnh, tracker làm 5 việc

1. **Đoán vị trí:**
   - Kalman đoán theo vận tốc ([04](04_kalman.md)).
   - Nếu Lucas-Kanade bám được đủ điểm trên người, độ dịch trung vị của các điểm được dùng để sửa lại dự đoán.
2. **Xem lại các box mới:**
   - Các mảnh **nằm chồng lên nhau theo chiều dọc** trong cùng một người (thân và chân bị đứt) thì gộp lại.
   - Một box chứa **từ 2 người trở lên** là cụm gộp; box đó bị bỏ ra khỏi bước ghép.
3. **Ghép lại người vừa tách khỏi cụm**, bằng 3 tín hiệu: còn gần chỗ Kalman đoán không, có giống ảnh mẫu chụp trước khi gộp không (NCC), và có giống màu áo không. Ghép bằng Hungarian.
4. **Ghép những người còn lại** với box sạch: Hungarian trên `1 − IoU` ([03](03_ghep_cap_hungarian.md)).
5. **Cập nhật trạng thái**, rồi tạo người mới từ box còn thừa.

## Máy trạng thái

| Trạng thái | Nghĩa | Ra khỏi trạng thái khi |
|---|---|---|
| `tentative` | Mới thấy, chưa chắc | Thấy 3 lần liên tiếp **và** đã đi được ≥ 5 px thì sang `confirmed`. Mất một lần thì bỏ |
| `confirmed` | Đã chắc chắn | Dính vào cụm → `merged`; mất box nhưng Lucas-Kanade cho thấy đứng yên → `static`; mất quá 10 frame → `lost` |
| `merged` | Đang dính với người khác | Tách ra và ghép lại được → `confirmed`; quá 90 frame → `lost` |
| `static` | Đứng yên, không còn box | Có box trở lại → `confirmed`; quá 300 frame → `lost` |
| `lost` | Kết thúc | ID không bao giờ dùng lại |

## Các quy tắc của tracker

1. **Khung khả nghi.** Nếu box mới to hoặc nhỏ bất thường (gấp quá 1,6 lần) **và** chồng lên người khác, thì vẫn coi là còn thấy người, nhưng **không lấy vị trí, kích thước, ảnh mẫu và màu** từ box đó. Không có quy tắc này, box của người đứng yên bị phình to bất thường khi có người đi ngang phía sau.
2. **Chỉ gộp mảnh khi chúng nằm trên dưới nhau.** Hai mảnh nằm ngang hàng thường là **hai người đi cạnh nhau**, không phải thân và chân của một người.
3. **Khi dính nhau, chỉ người đứng trước giữ Lucas-Kanade.** Người đứng trước là người có chân thấp nhất trong ảnh, tức gần camera nhất, nên vẫn được nhìn thấy đầy đủ. Người đứng sau bỏ hết điểm và đi theo vận tốc cũ, vì điểm của họ dễ trượt sang người đứng trước (XLA 2.4, 2.5).

4. **Ngưỡng có độ trễ khi ở trong cụm gộp.** Muốn **vào** cụm thì ≥ 60% box phải nằm trong cụm. Người **đã đang** trong cụm chỉ cần ≥ 30%. Trước đây người ở mép cụm cứ chập chờn "ra – vào", bị tính là mất dấu rồi bị xóa, và đó chính là mở đầu của các lần tráo tên.
5. **Chặn ghép lại khác màu.** Lúc vừa tách nhóm, track chỉ được nhận box có màu áo/quần đủ giống (≥ 0,5). Áo vàng không được nhận box áo đỏ.
6. **Chỉ tạo track mới từ khung đủ cao một người** (QĐ-27, rút ra trên video thật EPFL). Khung phải cao ít nhất 70% chiều cao người mong đợi (ban đầu 60%, nâng lên 70% khi hiệu chỉnh tham số, QĐ-29) ở vị trí chân đó (mô hình chiều cao, [10](10_tim_nguoi.md)), trừ khi khung chạm mép ảnh. Không có quy tắc này, mỗi mảnh đầu hoặc chân của người mặc áo đen lại sinh một track trùng, sống khoảng 1,4 giây rồi chết. Số ID trên EPFL phòng thí nghiệm 4 người giảm 199 → 85 (cam A) và 191 → 66 (cam B).

Các quy tắc 1–5 và các con số dùng để chọn chúng được rút ra khi thử trên video máy vẽ. Bảng thí nghiệm cũ đã gỡ
theo QĐ-31. Các quy tắc vẫn giữ trong code, nhưng **ngưỡng của chúng cần hiệu chỉnh lại trên video thật có đáp án**
([14](14_hieu_chinh_tham_so.md)). Các công tắc chỉ để làm thí nghiệm (`appearance_weight`, `appearance_update_rate`,
`template_search`) vẫn tắt.

## Kết quả trên video thật

Xem [12](12_he_thong.md). Video của nhóm chưa có đáp án nên chưa có MOTA, IDF1. Đếm được thì trong 1000 frame đầu
camera 0 có 7 tên cho tối đa 4 người, và một người được giữ tên liền 844 frame. Cả video thì mỗi camera có 26–75 tên
(sau khi nối tracklet).

## Khi nào hỏng (hạn chế cần ghi trong báo cáo)

- **Nhóm đi sát nhau ngay từ lúc xuất hiện:** chỉ có một track chung cho cả nhóm, đến khi họ tách ra.
- **Nhiều người đi ngược chiều trên cùng trục nhìn của camera** (hành lang nhìn dọc), hoặc nhiều người đứng sát nhau: chồng lên nhau tới vài giây, Kalman đoán lệch dần, và dễ tráo tên khi tách ra.
- **Đáp án EPFL là khung dựng lại từ vị trí chân** (rộng 0,4 × cao), kể cả khi người bị che. Box của tracker hẹp hơn hoặc chỉ ôm phần nhìn thấy thì IoU có thể dưới 0,5 và bị chấm là sai, dù tracker vẫn bám đúng người ([15](15_du_lieu_that_va_dap_an.md)).

## Thầy có thể hỏi

- *"Vì sao khung thấp không được tạo track mới?"* Vì khung thấp hơn nhiều so với một người đứng ở chỗ đó thường chỉ là mảnh (đầu hoặc chân) của người đã có track. Tạo track cho nó thì một người có hai tên. Khung đó vẫn được dùng để cập nhật track cũ; chỉ không sinh tên mới. Người đang bước vào ở mép ảnh thì được miễn.

- *"Hai người đi qua nhau thì sao?"* Tracker phát hiện cụm gộp trước khi ghép. Trong lúc dính nhau, người đứng trước vẫn bám bằng Lucas-Kanade, người đứng sau đi theo Kalman. Khi tách ra thì ghép lại bằng vị trí, ảnh mẫu và màu.
- *"Người đứng yên thì sao?"* Nền không cập nhật ở chỗ có người, nên người đó vẫn được phát hiện. Nếu vẫn mất box mà Lucas-Kanade cho thấy đứng yên, track chuyển sang `static` chứ không bị xóa.
