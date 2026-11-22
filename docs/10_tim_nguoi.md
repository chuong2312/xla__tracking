# 10. Tìm người trong ảnh (bước C)

> Code: `src/background.py` (mô hình nền), `src/detection.py` (tìm box). Tham số ở section `detection` của `config.yaml`.
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Ở mỗi tấm ảnh, tìm những vùng **đang có vật chuyển động** và có hình dạng giống người. Kết quả là các box. Bước này **chưa biết ai là ai**; việc đó do tracker làm.

## Cách làm

1. **Ảnh nền.** Lấy trung vị của 30 tấm đầu (1 giây cảnh trống), được ảnh nền. Hoặc dùng MOG2; chọn bằng `detection.background`.
2. **So với nền.** Pixel nào khác nền nhiều (độ dài vector chênh lệch 3 kênh màu > 30) là foreground, tức "vùng chuyển động".
3. **Khử bóng.** Bóng làm sàn tối đi nhưng gần như giữ màu. Pixel nào như vậy thì bỏ.
4. **Che vùng không cần xét** (ROI, vùng bỏ qua) ngay sau bước 2–3.
5. **Làm sạch mask:**
   - opening để xóa chấm nhiễu;
   - closing với kernel **dựng đứng 3×9**, để nối thân với chân mà ít nối hai người đứng cạnh nhau.
6. **Gom cụm** (connected components). Mỗi cụm là một **mảnh**.
7. **Gộp mảnh của cùng một người** (QĐ-26). Người mặc áo trùng màu nền bị đứt thành đầu và chân. Trước khi chạy, hệ thống tự ước lượng **chiều cao người mong đợi theo vị trí chân** từ các frame đầu (`src/height_model.py`). Hai mảnh xếp trên – dưới chỉ được gộp khi box gộp **không cao hơn một người**. Hai người đứng trước – sau gộp lại thì cao hơn hẳn, nên không bị gộp nhầm.
8. Lọc theo diện tích và tỉ lệ cao/rộng của người.
9. **Cập nhật nền có chọn lọc:** sau khi tracker chạy xong, nền chỉ được cập nhật ở những chỗ **không có người đang theo dõi**. Nhờ vậy người đứng yên không bị "nuốt" vào nền.

## Công thức

- Foreground: `‖I(x) − B(x)‖ > diff_threshold`.
- Bóng: `0,5 ≤ V_I / V_B ≤ 0,95`, `|S_I − S_B| ≤ 40` và `|H_I − H_B| ≤ 10`.
- Cập nhật nền: `B ← (1 − a)·B + a·I`, chỉ ở pixel không thuộc người. a = `update_rate`.
- Chiều cao mong đợi: `h(y) = a·y + b`, kẻ qua phân vị 90% chiều cao box trong từng khoảng 25 px của y_chân, bằng Theil–Sen: `a = trung vị((h_j − h_i)/(y_j − y_i))`, `b = trung vị(h_i − a·y_i)`.
- Minh họa: `python -m tools.explain_height_model --video data/real/4p-c1.mp4 --frame 1800` (ghi `outputs/explain/chieu_cao/`).
- Gộp hai mảnh khi: thẳng hàng dọc (chung ≥ 50% bề rộng), khoảng hở ≤ 1 × chiều cao mảnh cao hơn, và `cao_gộp ≤ 1,15 · h(y_chân_gộp)`.

## Kết quả trên video tổng hợp (precision / recall)

| Cảnh | Trừ nền trung vị | MOG2 |
|---|---|---|
| `s1` (đi riêng) | 1,00 / 0,99–1,00 | 1,00 / 0,99–1,00 |
| `s2`, `s3`, `hanh_lang_cua` (đi sát nhau) | 0,87–0,97 / 0,72–0,87 | 0,89–1,00 / 0,69–0,86 |

Hai cách cho kết quả gần nhau. Recall giảm ở các cảnh đông người vì **hai người dính thành một cụm** là điều không tránh được ở bước này; tracker phải xử lý chuyện đó. Nên chọn cách nào làm mặc định thì quyết định khi có video dev thật (QĐ-03).

## Trên video thật EPFL

- Trừ nền trung vị báo nhầm khoảng 15% ảnh (tường, thảm), do video không có cảnh trống ở đầu và ánh sáng nhấp nháy. MOG2 chỉ báo nhầm 5%, nên MOG2 là mặc định (QĐ-25).
- MOG2 cập nhật nền ở mọi chỗ, kể cả chỗ có người. Với learning rate 0,005, người đứng yên khoảng 8 giây là bị nuốt vào nền. Hạ xuống 0,001 (khoảng 40 giây) thì người đứng yên được giữ, và điểm tổng hợp tăng (QĐ-28).
- Người mặc áo đen trước tường tối bị mất phần thân. Gộp mảnh theo chiều cao (bước 7) giảm số box mỗi frame từ 4,6 xuống 3,5.

## Khi nào hỏng

- Camera rung hoặc trôi: cả ảnh khác nền.
- Hai người đi sát nhau: thành một cụm.
- Quần áo trùng màu nền: người bị "thủng", vỡ thành nhiều mảnh. Gộp theo chiều cao chữa được khi hai mảnh cách nhau không quá xa; mất hẳn cả người thì không chữa được. Hai người cùng áo đen chồng trước – sau có thể vẫn bị gộp nhầm.
- Sàn không phẳng hoặc camera nhìn ngang (không nhìn xuống): chiều cao không tăng đều theo vị trí chân, nên mô hình chiều cao sai.
- Phản chiếu trên kính, cửa đóng mở: báo nhầm.

## Thầy có thể hỏi

- *"Vì sao không cập nhật nền ở chỗ có người?"* Để người đứng yên lâu không bị coi là một phần của nền rồi biến mất.
- *"Vì sao kernel closing dựng đứng?"* Để nối thân với chân (nằm trên dưới nhau), mà không nối hai người đứng cạnh nhau (nằm ngang hàng).
- *"Mô hình chiều cao có phải học máy không?"* Không. Đó là thống kê: lấy phân vị 90% chiều cao trong từng dải vị trí chân, rồi kẻ một đường thẳng qua các điểm đó (Theil–Sen, dùng trung vị). Không có nhãn, không có bộ phân loại.
- *"Sao không gộp mọi mảnh nằm trên – dưới?"* Vì trong hành lang nhìn dọc, người ở xa nằm ngay trên đầu người ở gần. Đã thử và điểm giảm (QĐ-26). Chiều cao mới phân biệt được: hai mảnh của một người gộp lại vẫn cao bằng một người.
