# 09. Nhận dạng lại giữa hai camera (bước G)

> Code: `src/reid_similarity.py` (tính điểm giống nhau), `src/reid.py` (chuỗi xử lý), `tools/run_reid.py` (chạy và chấm điểm).
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

## Kết quả trên video tổng hợp (chế độ oracle: dùng khung đáp án)

| Kịch bản | Chỉ ngoại hình | Ngoại hình + sàn |
|---|---|---|
| `s1`, `s2`, `s3`, `door`, `hanh_lang_cua` | 4/4 | 4/4 |
| `s4` (người 1 và 2 mặc giống hệt) | **2/4**: người 1 và 2 bị tráo | **4/4** |

Ở `s4`, điểm keypoint và điểm màu của người 1 và người 2 gần như hòa nhau. Chỉ vị trí sàn phân biệt được (1,00 so với 0,04). Đây là lý do phải báo cáo riêng hai kết quả: để thấy rõ phần nào làm được việc gì.

**Đừng báo cáo con số 4/4 này như kết quả thật.** Video tổng hợp quá dễ: hai camera vẽ cùng một hoa văn, không có góc nghiêng thật, không đổi ánh sáng.

## Tham số (`config.yaml`, section `reid`)

| Tham số | Ý nghĩa |
|---|---|
| `match_threshold` (0,3) | Điểm tổng dưới mức này thì "không khớp ai" |
| `ratio_test` (0,75), `vertical_tolerance` (0,1) | Hai bộ lọc cặp khớp keypoint |
| `min_matches` (6), `feature_saturation_matches` (30) | Ngưỡng dưới và mức "đủ" của số cặp khớp |
| `use_ransac` (false) | Bật RANSAC chỉ khi làm thí nghiệm so sánh (QĐ-06) |
| `geo_sigma_px` (25) | Độ lệch cho phép của vị trí sàn |
| `weight_feature`, `weight_color`, `weight_geometry` | Trọng số khi gộp điểm |

Ngưỡng 0,3 được chọn trên video tổng hợp: cùng người có điểm 0,42–0,94, khác người tối đa 0,14. **Phải hiệu chỉnh lại trên video dev thật.**

## Chạy thử

```bash
python -m tools.run_reid --scene data/synthetic_v2/s4              # in số cặp đúng, rank-1, bảng điểm
python -m tools.run_reid --scene data/synthetic_v2/s4 --method orb # thử phương pháp khác
python -m tools.explain_reid --scene data/synthetic_v2/s4 --person 3
```

Ảnh `outputs/explain/nhan_dang_lai/s4_bang_diem.png` có 5 bảng điểm: viền đỏ là cặp đúng, viền xanh đứt là cặp chương trình chọn. Ảnh `s4_nguoi3_khop.png` nối các cặp keypoint khớp tốt; các đường gần như nằm ngang chính là nhờ ràng buộc dọc.

Hiện mới có **chế độ oracle**, tức dùng khung đáp án. Chế độ end-to-end (dùng track thật) sẽ làm khi tracker mới xong (giai đoạn 2).

## Khi nào hỏng

- **Hai người mặc giống nhau:** cần vị trí sàn, hoặc thời gian qua cửa ([07](07_ghep_su_kien_cua.md)).
- **Hai camera lệch góc lớn:** thấy hai mặt khác nhau, keypoint khớp kém. Thử ASIFT.
- **Ánh sáng hai camera rất khác:** màu lệch dù đã cân bằng gray-world.
- **Tracker cắt một người thành nhiều track:** cần nối các đoạn track trước khi ghép (XLA 2.10, làm ở giai đoạn sau).

## Thầy có thể hỏi

- *"Làm sao biết người ở camera A là ai ở camera B?"* So ba loại điểm (keypoint, màu, vị trí sàn), gộp lại, rồi ghép một-một bằng Hungarian; cặp điểm thấp thì để "không khớp ai".
- *"Vì sao báo cáo hai kết quả riêng?"* Vị trí sàn rất mạnh nhưng chỉ là bổ trợ và có thể sai. Báo riêng kết quả "chỉ ngoại hình" để chứng minh phần nhận dạng bằng đặc trưng ảnh làm được đến đâu.
