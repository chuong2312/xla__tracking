# 13. Nối các đoạn track bị đứt (tracklet linking)

> Code: `src/tracking/tracklet_linking.py`. Chạy tự động trong `run.py`. Tham số ở section `tracklet_linking` của `config.yaml`.
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Khi tracker mất dấu một người (bị che lâu, hoặc tách khỏi nhóm mà không ghép lại được), người đó xuất hiện lại với **một ID mới**. Cùng một người bị cắt thành hai, ba đoạn, nên phần nhận dạng lại giữa hai camera cũng bị ảnh hưởng (XLA 2.10).

Bước này chạy **sau khi** đã bám người xong cả video, và nối các đoạn của cùng một người lại thành một.

## Cách làm

Với mỗi cặp (đoạn trước i, đoạn sau j), xét 3 điều kiện:
1. **Thời gian:** j phải bắt đầu **sau khi** i kết thúc, cách không quá `max_gap_frames` frame. Hai đoạn chồng thời gian thì không thể là một người.
2. **Vị trí:** lấy vận tốc ở vài frame cuối của i để đoán i đã đi tới đâu lúc j bắt đầu. Chỗ bắt đầu của j phải gần chỗ đoán đó.
3. **Màu áo/quần:** phải giống nhau đủ mức.

Cặp nào qua cả 3 điều kiện thì được tính chi phí. Hungarian ghép một-một: mỗi đoạn nối với tối đa một đoạn sau. Các cặp nối thành chuỗi i → j → k, cả chuỗi lấy ID của đoạn đầu.

## Công thức

- Vị trí đoán: `p̂ = tâm cuối của i + (s_j − e_i) · vận tốc cuối của i`.
- Khoảng cách chuẩn hóa: `d = ‖p̂ − tâm đầu của j‖ / đường chéo box`, cần `d ≤ max_dist_ratio`.
- Chi phí: `0,5 · d + 0,5 · (1 − độ giống màu)`, cần `≤ max_cost`.

## Tham số

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `max_gap_frames` | 120 | Khoảng hở tối đa giữa hai đoạn (4 giây ở 30 fps) |
| `velocity_frames` | 5 | Đo vận tốc cuối trên 5 frame cuối |
| `max_dist_ratio` | 3,0 | Chỗ bắt đầu lệch khỏi chỗ đoán không quá 3 lần đường chéo box |
| `min_color_similarity` | 0,5 | Độ giống màu tối thiểu |
| `weight_motion`, `weight_color`, `max_cost` | 0,5; 0,5; 1,0 | Trọng số và chi phí tối đa |

### Hiệu chỉnh bằng quét lưới (10 lượt chạy: 5 cảnh hành lang × 2 camera)

| Cấu hình | IDF1 trung bình | Tổng số lần đổi tên nhầm | Số lần nối |
|---|---|---|---|
| Không nối | 0,783 | 13 | 0 |
| Khoảng hở 60, khoảng cách 1,5 | 0,788 | 12 | 1 |
| Khoảng hở 120, khoảng cách 1,5 | 0,788 | 12 | 1 |
| **Khoảng hở 120, khoảng cách 3 (chọn)** | **0,796** | **11** | 2 |

Bước nối giúp được, nhưng **ít**: trong hành lang, lỗi chính là **tráo tên** (cả hai track vẫn sống, chỉ đổi người), không phải track bị **đứt**. Muốn sửa tráo tên thì phải sửa ở tracker ([11](11_tracker_moi.md)).

## Khi nào hỏng

- **Hai người mặc giống nhau:** màu không phân biệt được, dễ nối nhầm người.
- **Người đổi hướng trong lúc bị mất dấu:** vị trí đoán theo vận tốc cũ bị lệch.
- **Tráo tên:** bước này không sửa được, vì không có đoạn nào bị đứt.

## Thầy có thể hỏi

- *"Vì sao cần nối tracklet?"* Để một người bị mất dấu rồi xuất hiện lại vẫn giữ một ID, nhờ vậy bước nhận dạng lại giữa hai camera và điểm IDF1 đúng hơn.
- *"Làm sao biết hai đoạn là cùng một người?"* Đoạn sau phải bắt đầu sau đoạn trước, ở gần chỗ đoán theo vận tốc, và có màu áo giống. Hungarian chọn cách nối tốt nhất cho tất cả các đoạn cùng lúc.
