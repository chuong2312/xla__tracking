# 04. Bộ lọc Kalman

> Code: `src/common/kalman.py`. Ảnh minh họa: `python -m tools.explain_kalman`.
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Khi một người bị người khác che, chương trình không còn thấy họ, tức không có phép đo nào. Nhưng người đó vẫn đang đi. Ta cần **đoán tiếp vị trí** để khi họ hiện ra lại, chương trình nhận ra đúng là họ (XLA 2.4). Ngoài ra, mỗi phép đo đều hơi rung, nên cần làm mượt.

## Ý tưởng bằng lời

Một chiếc xe đang chạy 10 m mỗi giây bị cây che mất 1 giây. Ta đoán xe đã đi thêm khoảng 10 m. Khi thấy xe trở lại, ta sửa lại vị trí theo cái thấy được, nhưng không tin tuyệt đối vào lần nhìn đó, vì mắt cũng có thể nhìn sai một chút.

Kalman làm đúng như vậy, mỗi frame hai bước:
1. **Dự đoán:** vị trí mới = vị trí cũ + vận tốc.
2. **Hiệu chỉnh:** khi có phép đo, lấy **trung bình có trọng số** giữa vị trí dự đoán và vị trí đo được. Nguồn nào ít nhiễu hơn thì được tin nhiều hơn.

Khi người bị che thì chỉ làm bước 1.

## Công thức

Trạng thái `x = [cx, cy, vx, vy]`: tâm của người (pixel) và vận tốc (pixel mỗi frame).

- **Dự đoán**, với Δt = 1 frame:
  - `x ← F·x`, với `F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]`, nghĩa là `cx ← cx + vx`, `cy ← cy + vy`, vận tốc giữ nguyên.
  - Độ không chắc chắn tăng lên: `P ← F·P·Fᵀ + Q`.
- **Hiệu chỉnh** bằng phép đo `z = [cx, cy]`:
  - `K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹`. K gọi là "hệ số tin": nằm giữa 0 (không tin phép đo) và 1 (tin hoàn toàn).
  - `x ← x + K·(z − H·x)`, `P ← (I − K·H)·P`.
- `Q` là nhiễu chuyển động: người có thể tăng tốc hoặc rẽ bất ngờ. Với mỗi trục, `Q = σa² · [[1/4, 1/2], [1/2, 1]]`, trong đó σa là độ lệch chuẩn của gia tốc.
- `R = σ²·I` là nhiễu đo. Có hai mức: đo bằng detection (`detection_noise_px`) và đo bằng Lucas-Kanade (`lk_noise_px`).

Nhóm dùng `cv2.KalmanFilter` (thầy cho dùng thư viện) nhưng tự đặt các ma trận F, H, Q, R.

## Tham số (`config.yaml`, section `kalman`)

| Tham số | Ý nghĩa | Giá trị |
|---|---|---|
| `acceleration_std_px` | Gia tốc ngẫu nhiên (pixel/frame²): người hay rẽ, đổi tốc thì tăng | 0.1 |
| `detection_noise_px` | Độ rung của tâm box phát hiện (pixel) | 4.0 |
| `lk_noise_px` | Độ rung của tâm đo bằng Lucas-Kanade (pixel) | 3.0 |
| `initial_velocity_std_px` | Độ không chắc chắn của vận tốc lúc mới tạo track | 5.0 |

### Ví dụ hiệu chỉnh tham số bằng quét lưới

Thử 5 giá trị của `acceleration_std_px` trên quỹ đạo của `s1`, `s2`, `s4` (camera A, phép đo nhiễu 4 px, cố ý che 30 frame):

| `acceleration_std_px` | Lệch ở cuối đoạn bị che | Lệch khi vẫn thấy người |
|---|---|---|
| 0,5 | 24,7 px | 2,76 px |
| 0,2 | 9,5 px | 2,24 px |
| **0,1 (chọn)** | **6,9 px** | **1,91 px** |
| 0,05 | 5,7 px | 1,67 px |
| 0,02 | 5,1 px | 1,60 px |

Giá trị nhỏ hơn cho điểm tốt hơn, vì người trong video tổng hợp luôn đi thẳng. Người thật hay rẽ; giá trị quá nhỏ làm Kalman "chậm hiểu" khi người đổi hướng. Vì vậy nhóm chọn 0,1 làm điểm khởi đầu, và **phải hiệu chỉnh lại trên video dev thật**. Đây là "hiệu chỉnh tham số", không phải "huấn luyện".

## Ảnh minh họa

`python -m tools.explain_kalman` sinh ảnh `outputs/explain/kalman/s2_nguoi1_cam_a.png`:
- **Ảnh trên:** đường đi thật (trắng), phép đo có nhiễu (chấm cam), Kalman ước lượng (xanh dương), và đoạn bị che (đỏ), lúc này Kalman chỉ dự đoán.
- **Ảnh dưới:** sai lệch so với vị trí thật theo thời gian. Khi còn thấy người, Kalman (xanh) lệch ít hơn phép đo (cam). Trong đoạn bị che, sai lệch tăng dần; sau 30 frame (1 giây) lệch khoảng 9 px.

## Khi nào hỏng

- **Người rẽ hoặc dừng trong lúc bị che:** Kalman vẫn đi thẳng theo vận tốc cũ.
- **Che quá lâu:** sai lệch tăng dần theo thời gian.
- **Vài phép đo sai ngay trước lúc bị che** (ví dụ box bị dính với người khác): vận tốc bị kéo lệch. Vì vậy tracker mới không dùng box bị gộp làm phép đo (XLA 2.3).

## Thầy có thể hỏi

- *"Kalman có phải học máy không?"* Không. Kalman là bộ ước lượng với công thức cố định; các ma trận do nhóm tự đặt, không học từ dữ liệu.
- *"Vì sao dùng hai mức nhiễu đo?"* Đo bằng detection và đo bằng Lucas-Kanade có độ rung khác nhau. Nguồn nào chính xác hơn thì được tin nhiều hơn.
- *"Giá trị 0,1 chọn thế nào?"* Bằng quét lưới, xem bảng ở trên.
