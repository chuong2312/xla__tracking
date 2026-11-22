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

## Ba quy tắc rút ra khi thử trên video tổng hợp

1. **Khung khả nghi.** Nếu box mới to hoặc nhỏ bất thường (gấp quá 1,6 lần) **và** chồng lên người khác, thì vẫn coi là còn thấy người, nhưng **không lấy vị trí, kích thước, ảnh mẫu và màu** từ box đó. Không có quy tắc này, box của người đứng yên bị phình từ rộng 32 lên 253 pixel khi có người đi ngang phía sau.
2. **Chỉ gộp mảnh khi chúng nằm trên dưới nhau.** Hai mảnh nằm ngang hàng thường là **hai người đi cạnh nhau**, không phải thân và chân của một người.
3. **Khi dính nhau, chỉ người đứng trước giữ Lucas-Kanade.** Người đứng trước là người có chân thấp nhất trong ảnh, tức gần camera nhất, nên vẫn được nhìn thấy đầy đủ. Người đứng sau bỏ hết điểm và đi theo vận tốc cũ, vì điểm của họ dễ trượt sang người đứng trước (XLA 2.4, 2.5).

4. **Ngưỡng có độ trễ khi ở trong cụm gộp.** Muốn **vào** cụm thì ≥ 60% box phải nằm trong cụm. Người **đã đang** trong cụm chỉ cần ≥ 30%. Trước đây người ở mép cụm cứ chập chờn "ra – vào", bị tính là mất dấu rồi bị xóa, và đó chính là mở đầu của các lần tráo tên.
5. **Chặn ghép lại khác màu.** Lúc vừa tách nhóm, track chỉ được nhận box có màu áo/quần đủ giống (≥ 0,5). Áo vàng không được nhận box áo đỏ.
6. **Chỉ tạo track mới từ khung đủ cao một người** (QĐ-27, rút ra trên video thật EPFL). Khung phải cao ít nhất 60% chiều cao người mong đợi ở vị trí chân đó (mô hình chiều cao, [10](10_tim_nguoi.md)), trừ khi khung chạm mép ảnh. Không có quy tắc này, mỗi mảnh đầu hoặc chân của người mặc áo đen lại sinh một track trùng, sống khoảng 1,4 giây rồi chết. Số ID trên EPFL giảm 199 → 85 (cam A) và 191 → 66 (cam B); điểm trên video tổng hợp không đổi.

### Thí nghiệm sửa lỗi tráo tên (10 lượt chạy: 5 cảnh hành lang × 2 camera)

| Thử | MOTA TB | IDF1 TB | Đổi tên nhầm | Giữ? |
|---|---|---|---|---|
| Trước khi sửa | 0,672 | 0,783 | 13 | |
| Thêm màu vào bước ghép bình thường (`appearance_weight` 0,5) | 0,665 | 0,779 | 14 | Không |
| Cập nhật màu và ảnh mẫu từ từ (`appearance_update_rate` 0,1) | 0,672 | 0,783 | 13 | Không (không đổi) |
| Ghép lại sau tách: ít vị trí, nhiều màu | 0,683 | 0,780 | 15 | Không |
| **Ngưỡng có độ trễ (`merge_keep_coverage` 0,3)** | **0,692** | **0,806** | 14 | **Có** |
| Chặn ghép lại khác màu (`recover_min_color_similarity` 0,5) | 0,670 | 0,782 | 13 | Có (an toàn, không hại) |
| Tìm lại người trong cụm bằng so khớp ảnh mẫu NCC (`template_search`), ngưỡng 0,4 / 0,6 / 0,75 | 0,662 / 0,619 / 0,668 | 0,781 / 0,763 / 0,794 | 14 / 15 / 11 | Không (tắt) |

Cách tìm ra nguyên nhân: soi từng frame lúc tách nhóm ở `s2`. Track của người 1 ở sát mép cụm, phần box trong cụm dao động 0,46–0,62 quanh ngưỡng 0,6, nên bị tính là mất dấu và bị xóa ở frame 205. Khi người 1 hiện ra, track của người 3 (vẫn còn khuất) "cướp" box của người 1. So khớp ảnh mẫu không giúp được, có lẽ vì ảnh mẫu chụp ở xa quá nhỏ (24×48), và người đứng sau bị che nên dễ khớp nhầm sang người đứng trước.

### Thí nghiệm so sánh cho quy tắc 3 (trung bình 12 lượt chạy trên bộ cảnh cũ, trước QĐ-22)

| Khi dính nhau thì xử lý Lucas-Kanade thế nào | MOTA | IDF1 |
|---|---|---|
| Không ai dùng | 0,742 | 0,81 |
| Người sau giữ các điểm ngoài vùng bị che | 0,714 | 0,80 |
| **Chỉ người trước dùng (chọn)** | **0,758** | **0,84** |

### Hai thử nghiệm KHÔNG cải thiện (bộ cảnh cũ, giữ để báo cáo)

| Thử | MOTA | IDF1 | Ghi chú |
|---|---|---|---|
| Kéo dài thời gian ghép lại bằng màu/ảnh mẫu sau khi tách nhóm: 1 → 5 → 15 frame (`recover_window_frames`) | 0,755 → 0,753 → 0,760 | 0,823 → 0,822 → 0,813 | Đổi tên nhầm tăng 8 → 10 → 14; giữ 1 |
| Chỉ chặn box to ra, không chặn box nhỏ đi (`guard_shrink`) | không đổi | không đổi | Giữ `true` |

## Kết quả trên video tổng hợp hành lang (tìm người + tracker mới, chưa nối tracklet)

| Cảnh | MOTA cam A / cam B | IDF1 cam A / cam B | Số lần đổi tên nhầm A / B |
|---|---|---|---|
| `s1` (đi thưa) | 0,86 / 0,84 | 0,93 / 0,85 | 0 / 1 |
| `s2` (cắt nhau) | 0,44 / 0,58 | 0,64 / 0,65 | 3 / 4 |
| `s3` (đứng yên) | 0,94 / 0,75 | 0,97 / 0,88 | 0 / 0 |
| `s4` (mặc giống) | 0,32 / 0,69 | 0,57 / 0,78 | 3 / 3 |
| `s5` (đi thành nhóm) | 0,69 / 0,83 | 0,85 / 0,92 | 0 / 0 |

Kết quả sau khi nối tracklet nằm ở [12](12_he_thong.md) và [13](13_noi_tracklet.md).

## Khi nào hỏng (hạn chế cần ghi trong báo cáo)

- **Nhóm đi sát nhau ngay từ lúc xuất hiện:** chỉ có một track chung cho cả nhóm, đến khi họ tách ra.
- **Nhiều người đi ngược chiều trên cùng trục nhìn của camera** (hành lang nhìn dọc, `s2`, `s4`): chồng lên nhau tới vài giây, Kalman đoán lệch dần, và dễ tráo tên khi tách ra.
- **Khi người bị che, đáp án chỉ bao phần nhìn thấy, còn box của tracker bao cả người.** Vì vậy IoU có thể dưới 0,5 và bị chấm là sai, dù tracker vẫn bám đúng người.

## Thầy có thể hỏi

- *"Vì sao khung thấp không được tạo track mới?"* Vì khung thấp hơn nhiều so với một người đứng ở chỗ đó thường chỉ là mảnh (đầu hoặc chân) của người đã có track. Tạo track cho nó thì một người có hai tên. Khung đó vẫn được dùng để cập nhật track cũ; chỉ không sinh tên mới. Người đang bước vào ở mép ảnh thì được miễn.

- *"Hai người đi qua nhau thì sao?"* Tracker phát hiện cụm gộp trước khi ghép. Trong lúc dính nhau, người đứng trước vẫn bám bằng Lucas-Kanade, người đứng sau đi theo Kalman. Khi tách ra thì ghép lại bằng vị trí, ảnh mẫu và màu.
- *"Người đứng yên thì sao?"* Nền không cập nhật ở chỗ có người, nên người đó vẫn được phát hiện. Nếu vẫn mất box mà Lucas-Kanade cho thấy đứng yên, track chuyển sang `static` chứ không bị xóa.
