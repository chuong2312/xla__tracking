# 11. Tracker mới: bám người và giữ tên (bước D)

> Code: `src/people_tracker.py` (điều phối), `src/track.py` (hồ sơ một người), `src/optical_flow.py` (Lucas-Kanade),
> `src/blob_analysis.py` (cụm gộp / tách), `src/recovery.py` (ghép lại sau khi tách). Dùng lại `src/kalman.py` và `src/assignment.py`.
> Tham số ở section `people_tracker` của `config.yaml`. Tracker cũ (`src/tracker.py`) được giữ lại để so sánh.

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

### Thí nghiệm so sánh cho quy tắc 3 (trung bình 12 lượt chạy: 6 cảnh × 2 camera)

| Khi dính nhau thì xử lý Lucas-Kanade thế nào | MOTA | IDF1 |
|---|---|---|
| Không ai dùng | 0,742 | 0,81 |
| Người sau giữ các điểm ngoài vùng bị che | 0,714 | 0,80 |
| **Chỉ người trước dùng (chọn)** | **0,758** | **0,84** |

## Kết quả trên video tổng hợp (tìm người + tracker mới)

| Cảnh | MOTA cam A / cam B | IDF1 cam A / cam B | Số lần đổi tên nhầm |
|---|---|---|---|
| `s1` | 0,94 / 0,94 | 0,97 / 0,97 | 0 |
| `s2` | 0,92 / 0,78 | 0,96 / 0,89 | 0 |
| `s3` (đứng yên) | 0,72 / 0,95 | 0,81 / 0,98 | 1 |
| `s4` (mặc giống) | 0,93 / 0,78 | 0,97 / 0,89 | 0 |
| `door` | 0,31 / 0,76 | 0,64 / 0,76 | 2 |
| `hanh_lang_cua` | 0,56 / 0,52 | 0,67 / 0,56 | 6 |

## Khi nào hỏng (hạn chế cần ghi trong báo cáo)

- **Nhóm đi sát nhau ngay từ lúc xuất hiện:** chỉ có một track chung cho cả nhóm, đến khi họ tách ra.
- **Hai người đi cùng một cột ảnh**, một người lại gần, một người đi xa (cảnh `door`, camera A): dính nhau lâu, Kalman đoán lệch dần.
- **Nhiều người chen nhau ở cửa** (cảnh `hanh_lang_cua`, camera B): bị tráo tên.
- **Khi người bị che, đáp án chỉ bao phần nhìn thấy, còn box của tracker bao cả người.** Vì vậy IoU có thể dưới 0,5 và bị chấm là sai, dù tracker vẫn bám đúng người. MOTA thấp ở `door cam_a` một phần do điều này.

## Thầy có thể hỏi

- *"Hai người đi qua nhau thì sao?"* Tracker phát hiện cụm gộp trước khi ghép. Trong lúc dính nhau, người đứng trước vẫn bám bằng Lucas-Kanade, người đứng sau đi theo Kalman. Khi tách ra thì ghép lại bằng vị trí, ảnh mẫu và màu.
- *"Người đứng yên thì sao?"* Nền không cập nhật ở chỗ có người, nên người đó vẫn được phát hiện. Nếu vẫn mất box mà Lucas-Kanade cho thấy đứng yên, track chuyển sang `static` chứ không bị xóa.
