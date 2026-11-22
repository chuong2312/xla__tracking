# 03. Ghép cặp bằng Hungarian

> Code: `src/assignment.py`. Ảnh minh họa: `python -m tools.explain_assignment`.
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Ở mỗi frame, chương trình có hai danh sách:
- các **track**: những người đang theo dõi, kèm vị trí dự đoán;
- các **box phát hiện** mới tìm được.

Phải ghép **mỗi track với tối đa một box**, và mỗi box với tối đa một track, sao cho hợp lý nhất. Bài toán y hệt cũng xuất hiện ở bước nhận dạng lại: ghép mỗi người ở camera A với một người ở camera B. Vì vậy hàm này dùng chung cho cả hai bước.

## Cách làm

1. Lập **bảng chi phí**: ô (i, j) là "độ không hợp" giữa track i và box j. Ở bước bám người, chi phí là `1 − IoU`: hai box càng chồng khít thì chi phí càng gần 0.
2. Cặp nào rõ ràng vô lý (IoU dưới ngưỡng) thì đánh dấu **cấm**.
3. Thuật toán Hungarian chọn các cặp một-một sao cho **tổng chi phí nhỏ nhất**.
4. **Lọc lại sau khi ghép:** bỏ cặp nào có chi phí vượt ngưỡng.

## Vì sao không ghép "tham lam"?

Ghép tham lam là cặp nào tốt nhất thì ghép trước. Ví dụ bảng chi phí:

|  | box 0 | box 1 |
|---|---|---|
| track 0 | 1 | 2 |
| track 1 | 2 | 100 |

- **Tham lam:** lấy (track 0, box 0) = 1 trước, nên track 1 buộc phải lấy box 1 = 100. Tổng 101.
- **Hungarian:** lấy (track 0, box 1) + (track 1, box 0) = 4. Tốt hơn hẳn.

Ví dụ này có trong `tests/test_assignment.py`.

## Hai chỗ dễ sai khi dùng hàm của SciPy (XLA 2.9)

1. `linear_sum_assignment` **luôn ghép đủ** số cặp có thể, kể cả cặp rất tệ. Vì vậy phải lọc theo ngưỡng **sau khi** ghép.
2. Nếu dùng "vô cùng" để đánh dấu cặp bị cấm mà có một hàng toàn vô cùng, hàm sẽ báo lỗi. Vì vậy cặp cấm được đổi thành một số rất lớn (`FORBIDDEN_COST = 1e6`), rồi bị loại ở bước lọc.

## Công thức

- `IoU(A, B) = diện tích phần chung / diện tích phần hợp`, nằm trong khoảng 0 đến 1.
- Chi phí bám người: `cost = 1 − IoU`, cấm khi `IoU < tracker.iou_gate`.
- Hungarian tìm cách ghép một-một làm `Σ cost(i, j)` nhỏ nhất. Độ phức tạp O(n³), với n là số người, nên rất nhanh vì n nhỏ.

## Tham số

| Tham số | Ý nghĩa |
|---|---|
| `max_cost` (đối số của hàm) | Cặp có chi phí lớn hơn bị bỏ sau khi ghép |
| `tracker.iou_gate` (0.1) | Dùng trong ảnh minh họa: IoU dưới mức này thì cấm ghép |

## Ảnh minh họa

`python -m tools.explain_assignment` sinh ảnh `outputs/explain/ghep_cap/s2_cam_b_frame174.png`:
- **Bên trái:** box dự đoán (nét đứt), box phát hiện (nét liền), và đường xanh nối các cặp được ghép.
- **Bên phải:** bảng IoU. Ô viền đỏ là cặp được chọn, ô "cấm" là cặp có IoU dưới ngưỡng.

Ở frame này, track 1 có tới 3 box được phép ghép (IoU 0,12; 0,42; 0,16). Hungarian chọn box có IoU 0,42 mà không làm hỏng cặp của người khác.

## Khi nào hỏng

- Hungarian chỉ tối ưu theo bảng chi phí mình đưa vào. Bảng chi phí sai (ví dụ hai người dính thành một box) thì kết quả cũng sai. Vì vậy tracker mới phải phân loại box bị gộp **trước khi** ghép (XLA 2.3).
- Nếu ngưỡng cấm quá chặt, người đi nhanh sẽ không ghép được với chính mình ở frame sau.

## Thầy có thể hỏi

- *"Hungarian giải bài toán gì?"* Bài toán phân công: ghép một-một sao cho tổng chi phí nhỏ nhất.
- *"Sao không dùng vô cùng cho cặp bị cấm?"* Xem mục "Hai chỗ dễ sai".
- *"Đây có phải học máy không?"* Không. Đây là tối ưu tổ hợp trên bảng chi phí tính bằng công thức.
