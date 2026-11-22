# 06. Chấm điểm bám người: MOTA, MOTP, IDF1

> Code: `src/evaluation/mot_metrics.py`. Chạy: `python -m src.evaluation.metrics track <tracks.csv> <gt_tracking.csv>`, lệnh in thêm bảng "MOT metrics".
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Khi bảo vệ, hội đồng thường hỏi "so với chuẩn nào?" (XLA 3.4). MOTA, MOTP và IDF1 là ba chỉ số chuẩn mà các bài báo về bám nhiều người đều dùng. Cả ba chỉ là **phép đếm** giữa kết quả của chương trình và đáp án; không có học máy.

## Cách chấm, theo từng frame có đáp án

1. Ghép người trong đáp án với track của chương trình. Một cặp khớp khi IoU ≥ `metrics.iou_threshold` (0,5).
   - Trước hết **giữ lại cặp của frame trước** nếu vẫn còn khớp, để không tính đổi tên chỉ vì cách ghép thay đổi một chút.
   - Phần còn lại ghép bằng Hungarian ([03_ghep_cap_hungarian.md](03_ghep_cap_hungarian.md)).
2. Đếm:
   - **FN (bỏ sót):** người có trong đáp án mà không track nào khớp.
   - **FP (báo nhầm):** track không khớp người nào.
   - **IDSW (đổi tên nhầm):** một người trong đáp án bị khớp với track khác so với lần khớp trước.
3. Chỉ dùng track ở trạng thái `confirmed`, giống `src/evaluation/metrics.py`.

## Công thức

- `MOTA = 1 − (FN + FP + IDSW) / GT`, với GT là tổng số box trong đáp án. Tối đa là 1; nếu chương trình sai nhiều hơn số người thật thì có thể âm.
- `MOTP = trung bình IoU của các cặp khớp`: khung tìm được sát với đáp án đến đâu.
- `IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)`:
  - Mỗi người trong đáp án được ghép với **đúng một track cho cả video**. Đây là ghép một-một toàn cục, chọn cách ghép có nhiều frame trùng nhất.
  - IDTP là số frame mang đúng tên. IDFN là số box trong đáp án không mang đúng tên. IDFP là số box của chương trình không mang đúng tên.

## MOTA và IDF1 khác nhau thế nào?

- **MOTA nặng về "tìm có đủ người không".** Đổi tên nhầm chỉ bị trừ 1 lần mỗi lần đổi.
- **IDF1 nặng về "giữ đúng tên suốt video".** Ví dụ trong `tests/test_mot_metrics.py`: hai người bị tráo tên đúng giữa video, MOTA vẫn đạt 0,9 nhưng IDF1 chỉ còn 0,5.

Đề tài coi trọng việc giữ đúng tên (để nhận dạng lại giữa hai camera), nên phải báo cáo **cả hai**.

## Kiểm tra nhanh

Lấy đáp án làm "kết quả" rồi chấm so với chính nó: phải ra MOTA = 1, MOTP = 1, IDF1 = 1. Đã thử trên `data/synthetic_v2/s2/cam_a` và ra đúng như vậy.

## Khi nào số liệu gây hiểu nhầm

- **Ngưỡng IoU 0,5 với người bị che:** box của chương trình thường bao cả người, còn đáp án chỉ bao phần nhìn thấy, nên có thể không đạt 0,5. Cần đọc kèm cột `occluded` của đáp án.
- **Chỉ chấm frame có đáp án:** nếu gán tay mỗi 10 frame và nội suy ở giữa, thì các frame nội suy cũng được dùng để chấm tracking. Riêng precision/recall của detector chỉ chấm keyframe (XLA 3.3).

## Thầy có thể hỏi

- *"MOTA âm nghĩa là gì?"* Tổng số lỗi nhiều hơn số người thật. Thường do báo nhầm rất nhiều.
- *"Sao cần IDF1 khi đã có MOTA?"* Vì MOTA gần như không phạt việc tráo tên kéo dài, trong khi đề tài cần giữ đúng tên.
