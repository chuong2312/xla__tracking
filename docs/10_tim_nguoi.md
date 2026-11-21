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
6. **Gom cụm** (connected components), rồi lọc theo diện tích và tỉ lệ cao/rộng của người.
7. **Cập nhật nền có chọn lọc:** sau khi tracker chạy xong, nền chỉ được cập nhật ở những chỗ **không có người đang theo dõi**. Nhờ vậy người đứng yên không bị "nuốt" vào nền.

## Công thức

- Foreground: `‖I(x) − B(x)‖ > diff_threshold`.
- Bóng: `0,5 ≤ V_I / V_B ≤ 0,95`, `|S_I − S_B| ≤ 40` và `|H_I − H_B| ≤ 10`.
- Cập nhật nền: `B ← (1 − a)·B + a·I`, chỉ ở pixel không thuộc người. a = `update_rate`.

## Kết quả trên video tổng hợp (precision / recall)

| Cảnh | Trừ nền trung vị | MOG2 |
|---|---|---|
| `s1` (đi riêng) | 1,00 / 0,99–1,00 | 1,00 / 0,99–1,00 |
| `s2`, `s3`, `hanh_lang_cua` (đi sát nhau) | 0,87–0,97 / 0,72–0,87 | 0,89–1,00 / 0,69–0,86 |

Hai cách cho kết quả gần nhau. Recall giảm ở các cảnh đông người vì **hai người dính thành một cụm** là điều không tránh được ở bước này; tracker phải xử lý chuyện đó. Nên chọn cách nào làm mặc định thì quyết định khi có video dev thật (QĐ-03).

## Khi nào hỏng

- Camera rung hoặc trôi: cả ảnh khác nền.
- Hai người đi sát nhau: thành một cụm.
- Quần áo trùng màu nền: người bị "thủng", vỡ thành nhiều mảnh.
- Phản chiếu trên kính, cửa đóng mở: báo nhầm.

## Thầy có thể hỏi

- *"Vì sao không cập nhật nền ở chỗ có người?"* Để người đứng yên lâu không bị coi là một phần của nền rồi biến mất.
- *"Vì sao kernel closing dựng đứng?"* Để nối thân với chân (nằm trên dưới nhau), mà không nối hai người đứng cạnh nhau (nằm ngang hàng).
