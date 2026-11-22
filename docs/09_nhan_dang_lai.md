# 09. Nhận dạng lại giữa hai camera (bước G)

> Code: `src/reid/reid_similarity.py` (tính điểm giống nhau), `src/reid/reid.py` (chuỗi xử lý), `tools/run_reid.py` (chạy và chấm điểm).
> Ảnh minh họa: `python -m tools.explain_reid`. Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Mỗi camera tự đặt số cho từng người. Bước này trả lời: **người số mấy ở camera A chính là người số mấy ở camera B?** Kết quả là một ID toàn cục chung cho cả hai camera.

## Cách làm

1. **Mô tả mỗi người** bằng ảnh mẫu, keypoint và màu ([08_dac_trung_ngoai_hinh.md](08_dac_trung_ngoai_hinh.md)).
2. **Tính 3 loại điểm giống nhau** cho mỗi cặp (người ở A, người ở B), mỗi loại từ 0 đến 1:

| Điểm | Cách tính |
|---|---|
| **Keypoint** | Khớp keypoint của hai người, chỉ giữ cặp khớp chắc chắn: tốt **hơn hẳn** cặp tốt nhì (ratio test), và **đầu khớp với đầu** (ràng buộc dọc). Dưới 6 cặp thì điểm = 0, đủ 30 cặp thì điểm = 1 |
| **Màu** | 1 − khoảng cách màu của thân trên và thân dưới |
| **Vị trí sàn** | Ở các tấm ảnh cả hai camera cùng thấy, quy đổi chỗ chân đứng từ A sang B bằng homography, xem có trùng người kia không. **Chỉ là thông tin bổ trợ** |

3. **Gộp điểm** = trung bình có trọng số. Có hai bảng, **luôn báo cáo riêng**:
   - **"chỉ ngoại hình"**: keypoint + màu;
   - **"ngoại hình + sàn"**: thêm vị trí sàn.
4. **Ghép bằng Hungarian** (module 03) sao cho tổng điểm cao nhất. Cặp có điểm dưới **ngưỡng 0,3** thì không ghép, và người đó nhận ID −1 ("không khớp ai").

## Công thức

- Điểm keypoint = `0` nếu n < 6, ngược lại `min(1, n / 30)`, với n là số cặp khớp tốt.
- Điểm vị trí sàn = trung bình `exp(−d² / 2σ²)`, với d là khoảng cách (pixel) giữa chân đã quy đổi và chân thật, σ = 25 px.
- Điểm tổng = `(0,4·keypoint + 0,4·màu + 0,2·sàn) / tổng trọng số của các điểm đang có`.

## Trên video của nhóm (QĐ-32)

Chế độ oracle (dùng khung đáp án làm track) cần đáp án. Bộ video EPFL 4 người chưa có đáp án, nên **chưa có số
người ghép đúng**. Đo được trên video, camera 0:
- **Màu áo:** cùng một người cách vài frame giống 0,82 và 0,86. Hai người khác nhau giống 0,64 (cả hai mặc áo đen)
  và 0,48. Màu phân biệt được người mặc khác màu, nhưng kém khi hai người cùng tông màu.
- **Điểm đặc trưng SIFT:** cùng một người cách 5–10 frame chỉ có **6–9 cặp khớp tốt**, điểm keypoint 0,20–0,30. Ảnh
  360×288 có ít chi tiết, nên phần keypoint yếu.
- Toàn hệ thống ghép được 20 cặp (c0 + c1) và 21 cặp (c2 + c3), nhưng chưa có đáp án để biết cặp nào đúng.

Vì ngoại hình yếu, **vị trí trên sàn** có thể là thông tin quan trọng nhất khi ghép hai camera. Điều này cần kiểm
chứng khi có đáp án. Hai kết quả "chỉ ngoại hình" và "ngoại hình + sàn" luôn được báo cáo riêng.

## Tham số (`config.yaml`, section `reid`)

| Tham số | Ý nghĩa |
|---|---|
| `match_threshold` (0,3) | Điểm tổng dưới mức này thì "không khớp ai" |
| `ratio_test` (0,75), `vertical_tolerance` (0,1) | Hai bộ lọc cặp khớp keypoint |
| `min_matches` (6), `feature_saturation_matches` (30) | Ngưỡng dưới và mức "đủ" của số cặp khớp |
| `use_ransac` (false) | Bật RANSAC chỉ khi làm thí nghiệm so sánh (QĐ-06) |
| `geo_sigma_px` (25) | Độ lệch cho phép của vị trí sàn |
| `weight_feature`, `weight_color`, `weight_geometry` | Trọng số khi gộp điểm |

Ngưỡng 0,3 và các trọng số là giá trị cũ, chọn trước QĐ-31, nên **phải hiệu chỉnh lại khi video của nhóm có đáp án** ([14](14_hieu_chinh_tham_so.md)).

## Chạy thử

```bash
# Cần cảnh CÓ ĐÁP ÁN (cam_a/gt_tracking.csv, cam_b/gt_tracking.csv, gt_reid.csv), xem [15](15_du_lieu_that_va_dap_an.md)
python -m tools.run_reid --scene data/canh/<cảnh có đáp án>                # in số cặp đúng, rank-1, bảng điểm
python -m tools.run_reid --scene data/canh/<cảnh có đáp án> --method orb   # thử phương pháp khác
python -m tools.explain_reid --scene data/canh/<cảnh có đáp án> --person 1
```

Ảnh trong `outputs/explain/nhan_dang_lai/` gồm 5 bảng điểm: viền đỏ là cặp đúng, viền xanh đứt là cặp chương trình chọn.
Ảnh nối cặp keypoint cho thấy các cặp khớp tốt; các đường gần như nằm ngang là nhờ ràng buộc dọc.

Toàn hệ thống (dùng track thật thay cho khung đáp án) chạy bằng `run.py`; kết quả ở [12](12_he_thong.md).
