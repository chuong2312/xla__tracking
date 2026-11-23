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

### Hiệu chỉnh tham số

`acceleration_std_px` = 0,1 là giá trị cũ, chọn trước QĐ-31. Giá trị nhỏ làm Kalman đoán mượt khi người đi thẳng nhưng
"chậm hiểu" khi người rẽ; giá trị lớn thì ngược lại. **Phải hiệu chỉnh lại trên video thật có đáp án**
(`python -m tools.hieu_chinh --set "kalman.acceleration_std_px=[0.05, 0.1, 0.2, 0.5]"`). Đây là "hiệu chỉnh tham số",
không phải "huấn luyện".

## Ảnh minh họa (video thật)

```bash
python run.py --scene data/canh/epfl_lab4p_c0_c1
python -m tools.explain_kalman --run outputs/he_thong/epfl_lab4p_c0_c1 --scene data/canh/epfl_lab4p_c0_c1 --camera cam_a
```

Công cụ lấy đoạn dài nhất mà một người được bám liên tục, cố ý giấu phép đo 30 frame ở giữa (như thể người bị che), rồi
chạy Kalman (`outputs/explain/kalman/`):
- **Ảnh trên:** vị trí đo được trên video thật (trắng), Kalman ước lượng (xanh dương), đoạn giấu phép đo (đỏ), lúc này Kalman chỉ dự đoán.
- **Ảnh dưới:** Kalman lệch bao nhiêu pixel so với vị trí đo được. Trên camera 0 của video nhóm (track dài nhất), sau
  30 frame (1,2 giây) không có phép đo, Kalman lệch khoảng 47 px: người thật đổi hướng và đổi tốc độ, không đi thẳng đều.
