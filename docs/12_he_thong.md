# 12. Toàn bộ hệ thống: `run.py`

> Code: `run.py` (lệnh chạy), `src/pipeline.py` (một camera), `src/tracking/tracklet_linking.py` (nối track bị đứt),
> `src/evaluation/system_eval.py` (chấm điểm toàn hệ thống). Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Chạy

```bash
python run.py --scene data/canh/epfl_lab4p_c0_c1 --debug-video
python run.py --scene data/canh/epfl_lab4p_c0_c1 --max-frames 300     # chạy thử nhanh 300 frame đầu
```

Lệnh này chạy trên một thư mục cảnh có `cam_a/video.<đuôi>` (và `cam_b/video.<đuôi>` nếu có camera thứ hai), xem
[data/README.md](../data/README.md). Kết quả ghi vào `outputs/he_thong/<cảnh>/`. Một cảnh dài khoảng 2,6 phút
với 2 camera mất khoảng 3–5 phút. Chỉ dùng video thật của nhóm (QĐ-31, QĐ-32).

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

## Kết quả trên video của nhóm (QĐ-32)

Video hiện có là bộ EPFL phòng thí nghiệm 4 người, 4 camera, **chưa có đáp án**. Vì vậy chưa tính được MOTA,
IDF1 hay số người ghép đúng; chỉ đếm được (`outputs/he_thong/<cảnh>/`, chạy ngày 2026-11-23):

| Cảnh | Camera | Khung tìm được mỗi frame | Track mỗi frame | Số ID: gốc → sau khi nối | Người giữ tên lâu nhất |
|---|---|---|---|---|---|
| `epfl_lab4p_c0_c1` | A / B | 3,15 / 3,38 | 3,24 / 3,19 | 95 → 42 / 81 → 26 | 1904 / 1523 frame |
| `epfl_lab4p_c2_c3` | A / B | 3,01 / 2,97 | 3,17 / 3,00 | 111 → 56 / 156 → 75 | 1438 / 1551 frame |

Ghép người giữa hai camera (ngoại hình + vị trí sàn): 20 cặp ở c0 + c1, 21 cặp ở c2 + c3. Chưa có đáp án nên
chưa biết cặp nào đúng.

**Đọc kết quả thế nào:**
- Cảnh có tối đa 4 người mà mỗi camera vẫn có 26–75 tên. Một người bị tách thành nhiều đoạn track: khi đi qua
  nhau, khi mặc đồ tối trước nền tối, khi ra vào mép khung. Nối tracklet đã gộp được khoảng một nửa số đoạn.
- Có người được giữ tên liền tới 1500–1900 frame, tức hơn 1 phút.
- Những gì đo được mà không cần đáp án (khung đúng cỡ người, màu áo…) ở [15](15_du_lieu_that_va_dap_an.md).

## Hướng cải thiện tiếp

1. **Khoanh đáp án** cho một đoạn video của nhóm (`tools/annotate.py`, [15](15_du_lieu_that_va_dap_an.md)), để chấm
   được MOTA, IDF1 và nhận dạng lại thật.
2. **Hiệu chỉnh lại tham số** khi đã có đáp án (`python -m tools.hieu_chinh`, [14](14_hieu_chinh_tham_so.md)). Nhiều
   ngưỡng được chọn trên video máy vẽ trước đây.
3. **Ghép người giữa hai camera dựa nhiều hơn vào vị trí sàn**: ở ảnh 360×288, điểm đặc trưng SIFT khớp ít (6–9 cặp
   cho cùng một người) và màu áo phân biệt kém.
4. **Giữ đúng tên khi nhiều người chồng lên nhau lâu.**
