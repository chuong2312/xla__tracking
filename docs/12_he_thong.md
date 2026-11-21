# 12. Toàn bộ hệ thống: `run.py`

> Code: `run.py` (lệnh chạy), `src/pipeline.py` (một camera), `src/system_eval.py` (chấm điểm toàn hệ thống).
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Chạy

```bash
python run.py --scene data/synthetic_v2/hanh_lang_cua --debug-video
```

Lệnh này chạy trên thư mục cảnh có `cam_a/video.mp4` và `cam_b/video.mp4`. Trên máy nhóm, một cảnh 12 giây với 2 camera mất khoảng 9 giây. Kết quả ghi vào `outputs/he_thong/<cảnh>/`.

## Hệ thống làm gì, theo thứ tự

```
Mỗi camera:  video → [C] tìm người → [D] bám người, giữ tên → [E] đếm VÀO/RA
                                    (src/pipeline.py)
Hai camera:  [F] mô tả ngoại hình mỗi track → [G] ghép người A ↔ B (src/reid.py)
             + ghép lượt qua cửa theo thời gian (src/door_matching.py)
Chấm điểm:   nếu có đáp án (src/system_eval.py)
```

## File kết quả

| File | Nội dung |
|---|---|
| `cam_a/detections.csv`, `cam_b/detections.csv` | Box tìm được ở từng tấm ảnh |
| `cam_a/tracks.csv`, `cam_b/tracks.csv` | Track của từng người: frame, ID, trạng thái, box |
| `cam_a/debug.mp4`, `cam_b/debug.mp4` | Video có vẽ box (màu theo trạng thái) và vạch cửa, khi chạy với `--debug-video` |
| `events.csv` | Các lượt qua cửa: `event_id,camera,frame,track_id,global_id,direction` |
| `global_ids_ngoai_hinh.csv`, `global_ids_ngoai_hinh_san.csv` | ID toàn cục: chỉ ngoại hình / ngoại hình + vị trí sàn |
| `door_links.csv` | Cặp track A ↔ B ghép được nhờ thời gian qua cửa |
| `summary.json` | Bảng điểm (khi có đáp án) |

## Cách chấm điểm toàn hệ thống (end-to-end)

Track do chương trình tự tạo, nên phải gắn mỗi track với một người trong đáp án trước:
1. Ở mỗi tấm ảnh, ghép track với người trong đáp án bằng Hungarian trên IoU.
2. Mỗi track được gắn với người mà nó khớp nhiều tấm ảnh nhất (**bỏ phiếu đa số**).
3. Nếu một người bị tách thành nhiều track, **track chính** là track khớp lâu nhất.

Sau đó:
- **Nhận dạng lại:** một người tính là đúng khi track chính của họ ở camera A và ở camera B có cùng ID toàn cục.
- **Đếm cửa:** một lượt tính là đúng khi đúng người, đúng chiều, và lệch không quá 1,5 giây so với đáp án.
- **Ghép qua cửa:** một cặp tính là đúng khi hai track được gắn với cùng một người.

## Kết quả trên video tổng hợp

| Cảnh | Đếm cửa (A, B) | Nhận dạng lại: ngoại hình / + sàn | Ghép qua cửa |
|---|---|---|---|
| `s1` | – | 4/4 / 4/4 | – |
| `s2` | 1/1, 1/1 | 4/4 / 4/4 | 1/1 |
| `s3` | – | 4/4 / 4/4 | – |
| `s4` | – | **2/4 / 4/4** | – |
| `door` | 3/4, 3/4 | 3/4 / 3/4 | 3/3 |
| `hanh_lang_cua` | 3/3, **1/3** | 2/4 / 2/4 | 1/2 |

Điểm bám người (MOTA, IDF1) từng cảnh nằm ở [11_tracker_moi.md](11_tracker_moi.md).

**Đọc kết quả thế nào:**
- Các cảnh thường (`s1`–`s4`) chạy tốt từ đầu đến cuối.
- `s4` cho thấy mặc giống nhau thì chỉ vị trí sàn mới cứu được (QĐ-02).
- Điểm yếu nhất là **nhiều người chen nhau ở cửa** (`hanh_lang_cua`, camera B): tracker bị tráo tên, kéo theo đếm cửa và nhận dạng lại sai.
- **Đây là video tổng hợp; không được báo cáo như kết quả thật.**

## Hướng cải thiện tiếp

- Nối các đoạn track bị đứt của cùng một người trước khi nhận dạng lại (XLA 2.10).
- Kết hợp kết quả ghép qua cửa vào ID toàn cục. Hiện hai kết quả đang tách riêng.
- Hiệu chỉnh tham số trên video dev thật.
