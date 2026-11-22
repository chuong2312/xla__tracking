# 12. Toàn bộ hệ thống: `run.py`

> Code: `run.py` (lệnh chạy), `src/pipeline.py` (một camera), `src/tracking/tracklet_linking.py` (nối track bị đứt),
> `src/evaluation/system_eval.py` (chấm điểm toàn hệ thống). Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Chạy

```bash
python run.py --scene data/synthetic_v2/s2 --debug-video
```

Lệnh này chạy trên thư mục cảnh có `cam_a/video.mp4` và `cam_b/video.mp4`. Kết quả ghi vào `outputs/he_thong/<cảnh>/`. Một cảnh 12 giây với 2 camera mất khoảng 10–15 giây.

## Hệ thống làm gì, theo thứ tự

```
Mỗi camera:  video → [C] tìm người → [D] bám người, giữ tên       (src/pipeline.py)
                   → nối các đoạn track bị đứt của cùng một người  (src/tracking/tracklet_linking.py)
Hai camera:  [F] mô tả ngoại hình mỗi track → [G] ghép người A ↔ B  (src/reid/reid.py)
Chấm điểm:   nếu có đáp án (src/evaluation/system_eval.py)
```

Phần đếm VÀO/RA qua cửa đã bị bỏ (QĐ-21).

## File kết quả

| File | Nội dung |
|---|---|
| `cam_a/detections.csv`, `cam_b/detections.csv` | Box tìm được ở từng tấm ảnh |
| `cam_a/tracks.csv`, `cam_b/tracks.csv` | Track gốc của từng người (frame, ID, trạng thái, box) |
| `cam_a/tracks_linked.csv`, `cam_b/tracks_linked.csv` | Track sau khi nối các đoạn bị đứt |
| `cam_a/debug.mp4`, `cam_b/debug.mp4` | Video có vẽ box, màu theo trạng thái (khi chạy với `--debug-video`) |
| `global_ids_ngoai_hinh.csv`, `global_ids_ngoai_hinh_san.csv` | ID toàn cục: chỉ ngoại hình / ngoại hình + vị trí sàn |
| `summary.json` | Bảng điểm (khi có đáp án) |

## Cách chấm điểm toàn hệ thống (end-to-end)

Track do chương trình tự tạo, nên phải gắn mỗi track với một người trong đáp án:
1. Ở mỗi tấm ảnh, ghép track với người trong đáp án bằng Hungarian trên IoU.
2. Mỗi track được gắn với người nó khớp nhiều tấm ảnh nhất (**bỏ phiếu đa số**).
3. Nếu một người bị tách thành nhiều track, **track chính** là track khớp lâu nhất.
4. **Nhận dạng lại** tính là đúng khi track chính của một người ở camera A và ở camera B có cùng ID toàn cục.

Bám người được chấm hai lần: trước và sau khi nối tracklet, để thấy bước nối giúp được bao nhiêu.

## Kết quả trên video tổng hợp hành lang

### Mặc định hiện tại: MOG2 quên chậm + gộp mảnh theo chiều cao (QĐ-25, QĐ-26, QĐ-27, QĐ-28)

| Cảnh | MOTA A / B (sau khi nối) | IDF1 A / B: trước → sau khi nối | Nhận dạng lại: ngoại hình / + sàn |
|---|---|---|---|
| `s1` (đi thưa) | 0,94 / 0,84 | 0,97 → 0,97 / 0,86 → 0,86 | 4/4 / 4/4 |
| `s2` (cắt nhau) | 0,40 / 0,51 | 0,44 → 0,51 / 0,62 → 0,72 | **2/4 / 2/4** |
| `s3` (đứng yên) | 0,93 / 0,82 | 0,97 → 0,97 / 0,86 → 0,86 | 4/4 / 4/4 |
| `s4` (mặc giống) | 0,56 / 0,68 | 0,58 → 0,77 / 0,78 → 0,78 | 4/4 / 4/4 |
| `s5` (đi thành nhóm) | 0,94 / 0,93 | 0,97 → 0,97 / 0,97 → 0,97 | 4/4 / 4/4 |

Trung bình 10 camera: MOTA 0,755, IDF1 0,836. So với trừ nền trung vị (bảng dưới, MOTA 0,683, IDF1 0,820): tốt hơn ở `s1`, `s3`, `s4`, `s5`; nhận dạng lại `s4` chỉ dùng ngoại hình đạt 4/4. Điểm yếu còn lại là `s2`: hai người đi ngược chiều cắt nhau, nhận dạng lại 2/4.

Các bước trung gian (trung bình MOTA / IDF1): MOG2 0,718 / 0,814 → + gộp mảnh theo chiều cao 0,723 / 0,822 → + chỉ tạo track từ khung đủ cao 0,723 / 0,822 → + quên nền chậm 0,755 / 0,836.

### Trước đó: trừ nền trung vị, chưa gộp mảnh

| Cảnh | MOTA A / B (sau khi nối) | IDF1 A / B: trước → sau khi nối | Nhận dạng lại: ngoại hình / + sàn |
|---|---|---|---|
| `s1` (đi thưa) | 0,86 / 0,84 | 0,93 → 0,93 / 0,85 → 0,85 | 4/4 / 4/4 |
| `s2` (cắt nhau) | 0,44 / 0,58 | 0,64 → 0,65 / **0,65 → 0,78** | 4/4 / 4/4 |
| `s3` (đứng yên) | 0,94 / 0,75 | 0,97 → 0,97 / 0,88 → 0,88 | 4/4 / 4/4 |
| `s4` (mặc giống) | 0,32 / 0,69 | **0,57 → 0,62** / 0,78 → 0,78 | **2/4 / 4/4** |
| `s5` (đi thành nhóm) | 0,69 / 0,83 | 0,85 → 0,85 / 0,92 → 0,92 | 4/4 / 4/4 |

Kết quả trên là sau khi sửa lỗi tráo tên (QĐ-24). Trước khi sửa: `s2` cam B IDF1 0,66, `s3` cam B 0,80, `s4` cam B 0,69, và `s4` nhận dạng lại chỉ 1/4 / 1/4.

Với khung đáp án (chế độ oracle), nhận dạng lại ghép đúng 4/4 ở **cả 5 cảnh**.

**Đọc kết quả thế nào:**
- Bản thân bước nhận dạng lại làm tốt: oracle đúng 4/4 ở mọi cảnh. Lỗi end-to-end ở `s4` (chỉ ngoại hình 2/4) đến từ **bước bám người** và từ việc hai người mặc giống nhau. Trong hành lang nhìn dọc, ba người đi ngược chiều chồng lên nhau khoảng 3,5 giây; tracker giữ trạng thái "đang dính" quá lâu, mất dấu, rồi tạo ID mới. Track chính của mỗi người vì thế không còn đúng.
- Nối tracklet giúp được ở `s3` và `s4`, nhưng không nhiều, vì lỗi chính là **tráo tên** (cả hai track vẫn sống, chỉ đổi người), không phải track bị **đứt**.
- `s1`, `s3 cam_a`, `s5` chạy tốt. `s2`, `s4` là điểm yếu: nhiều người đi ngược chiều trên cùng trục nhìn của camera.
- **Đây là video tổng hợp; không báo cáo như kết quả thật.**

## Trên video thật EPFL (`data/real/epfl_4p_c0_c1`)

```bash
python run.py --scene data/real/epfl_4p_c0_c1 --debug-video   # khoảng 2 phút
```

Chưa có đáp án nên chưa chấm điểm được, chỉ đếm (cảnh có tối đa 4 người):

| | Box mỗi frame A / B | Track đang chạy mỗi frame A / B | Số ID A / B (sau khi nối) |
|---|---|---|---|
| MOG2, chưa gộp mảnh | 4,6 / 4,6 | 5,9 / 5,9 | 200 / 203 |
| MOG2 + gộp mảnh theo chiều cao | 3,5 / 3,5 | 4,8 / 4,6 | 199 / 191 |
| + chỉ tạo track từ khung đủ cao (QĐ-27) | 3,5 / 3,5 | 3,6 / 3,3 | 85 / 66 |
| + MOG2 quên nền chậm (QĐ-28) | | 3,7 / 3,5 | **54 / 48** |

Số ID vẫn còn nhiều hơn nhiều so với 4 người, nhưng đã giảm hơn một nửa. Ảnh debug cho thấy hai lỗi còn lại:
- **Bỏ sót người:** cảnh có tới 4 người, nhưng mỗi frame chỉ có khoảng 2,6 khung đủ cao một người. Người đứng yên giờ đã được giữ (QĐ-28); người mặc đồ tối đi trước nền tối thì vẫn chỉ lộ cái đầu.
- **Vẫn còn vài track chỉ ôm cái đầu:** chúng được tạo lúc người ở sát mép ảnh (được miễn xét chiều cao), rồi sống tiếp. Muốn hiệu chỉnh tham số cho video thật thì cần đáp án, ít nhất ở một số frame mốc (`tools/annotate.py`).

**Nhận dạng lại trên EPFL** (chưa có đáp án nên chưa biết đúng hay sai, chỉ đếm): homography sàn lấy từ file hiệu chỉnh của EPFL ([02](02_homography_san.md)). Trong 102 track đủ dài, chỉ dùng ngoại hình ghép được 26 cặp A ↔ B; thêm vị trí sàn ghép được 29 cặp. Số cặp nhiều hơn 4 người vì mỗi người vẫn còn bị tách thành nhiều track.

Muốn chỉnh tham số cho hệ thống: xem [14](14_hieu_chinh_tham_so.md).

## Hướng cải thiện tiếp

- Giữ đúng tên khi nhiều người chồng lên nhau lâu (`s2`, `s4` camera A vẫn yếu). Đã thử dùng màu khi ghép bình thường và so khớp ảnh mẫu trong cụm, cả hai đều không giúp được ([11](11_tracker_moi.md)); cần cách khác, ví dụ tách cụm theo chiều sâu (ai đứng trước, ai đứng sau).
- Hiệu chỉnh tham số trên video dev thật.
