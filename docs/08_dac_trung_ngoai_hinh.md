# 08. Đặc trưng ngoại hình (bước F)

> Code: `src/reid/gallery.py` (chọn ảnh mẫu), `src/reid/keypoint_features.py` (keypoint), `src/reid/color_features.py` (màu).
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Muốn biết người ở camera A có phải người ở camera B không, trước hết phải **mô tả ngoại hình** của từng người bằng những con số so sánh được. Bước này làm việc đó, theo 3 phần:
- chọn ảnh mẫu;
- tìm điểm đặc trưng (keypoint);
- đo màu quần áo.

## 1. Chọn ảnh mẫu (gallery)

Một người xuất hiện ở hàng trăm tấm ảnh, nhưng chỉ cần vài tấm **đẹp nhất**. Một tấm được chọn khi:
- người **không bị che**;
- người **không chạm mép ảnh**, vì bị cắt thì thiếu một phần;
- người **to rõ**: ưu tiên khung lớn;
- các tấm **cách nhau một khoảng**, để có nhiều tư thế khác nhau.

Mặc định lấy tối đa 5 tấm mỗi người.

## 2. Keypoint (điểm đặc trưng)

1. **Cắt và chuẩn hóa:** cắt người ra, đưa về cùng một cỡ 128×256, giữ đúng tỉ lệ. Phần thừa đệm màu đen, và có một mask đánh dấu đâu là người thật.
2. **Tìm keypoint chỉ trên người:** mask (đã co vào trong vài pixel) được đưa thẳng cho hàm tìm keypoint. **Không tô đen nền lên ảnh**, vì tô đen tạo ra cạnh giả dọc viền người và sinh keypoint giả (XLA 2.6).
3. **Bốn phương pháp, giữ cùng số điểm** (tối đa 200 điểm mạnh nhất) để so sánh công bằng:

| Phương pháp | Ý tưởng |
|---|---|
| **SIFT** | Mặc định. Vẫn nhận ra điểm khi ảnh to, nhỏ hay xoay |
| **ORB** | Nhanh hơn, kém chính xác hơn. Đã chỉnh tham số cho crop nhỏ (XLA 2.7) |
| **sift_auto** | SIFT, nhưng nếu tìm được quá ít điểm thì tự hạ ngưỡng rồi chạy lại |
| **ASIFT** | SIFT thêm bước giả lập nhiều góc nghiêng, nên chịu được hai camera lệch góc nhau; chậm hơn |

## 3. Màu quần áo

1. **Cân bằng màu theo từng camera (gray-world):** hai điện thoại cho màu hơi khác nhau. Giả định màu trung bình của cả cảnh là màu xám, rồi chỉnh từng kênh màu cho đúng như vậy (XLA 2.12).
2. **Đếm màu riêng thân trên và thân dưới:** áo và quần thường khác màu. Bỏ phần đầu, vì đầu ai trông cũng gần giống nhau.
3. **So hai người:** độ giống màu = 1 − khoảng cách Bhattacharyya. Được 1 là giống hệt, 0 là khác hẳn.

## Công thức chính

- **Gray-world:** `gain_c = trung bình 3 kênh / trung bình kênh c`. Ảnh mới = ảnh × gain.
- **Độ giống màu** = `1 − ½·(Bhattacharyya(thân trên) + Bhattacharyya(thân dưới))`.

## Tham số (`config.yaml`, section `features`)

| Tham số | Ý nghĩa |
|---|---|
| `crop_size` [128, 256] | Cỡ crop chuẩn |
| `method` (sift) | Phương pháp keypoint mặc định |
| `max_keypoints` (200) | Số keypoint tối đa của mọi phương pháp |
| `mask_erode_px` (4) | Co mask vào trong bấy nhiêu pixel |
| `upper_body`, `lower_body` | Vùng thân trên và thân dưới, theo tỉ lệ chiều cao |
| `gallery_size` (5), `gallery_min_gap_frames` (15) | Số ảnh mẫu, và khoảng cách giữa hai ảnh mẫu |

## Khi nào hỏng

- **Quần áo trơn, không hoa văn:** gần như không có keypoint; chỉ còn màu.
- **Hoa văn lặp lại đều** (sọc, ô vuông): nhiều điểm giống hệt nhau nên khó khớp đúng. Trên video tổng hợp, người mặc áo sọc chỉ có khoảng 6 cặp khớp, trong khi áo ô vuông có tới 71.
- **Hai camera thấy hai mặt khác nhau** (trước/sau): logo, balo khác hẳn.
- **Hai người mặc giống nhau:** cả keypoint lẫn màu đều không phân biệt được (kịch bản S4).

## Thầy có thể hỏi

- *"Vì sao không tô đen nền rồi mới tìm keypoint?"* Vì sẽ sinh keypoint giả dọc viền người; dùng tham số `mask` thay vào đó.
- *"Có phải học máy không?"* Không. SIFT, ORB và histogram màu đều là đặc trưng thiết kế sẵn, không học từ dữ liệu.
