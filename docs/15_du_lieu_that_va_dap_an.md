# 15. Dữ liệu thật và đáp án

> Code: `tools/tao_canh.py` (tạo thư mục cảnh), `tools/epfl_homography.py` (phép chuyển điểm sàn giữa hai camera),
> `tools/annotate.py` và `tools/verify_gt.py` (nhóm tự khoanh và kiểm tra đáp án). Hướng dẫn đặt video:
> [data/README.md](../data/README.md). Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Dữ liệu hiện có

Dự án **chỉ dùng video thật** (QĐ-31) và **chỉ dùng video do nhóm đưa** (QĐ-32). Hiện có một bộ: **EPFL phòng
thí nghiệm 4 người**, 4 camera quay cùng lúc, mỗi video 360×288, 25 fps, 3915 frame. Hai cảnh để chạy:
- `data/canh/epfl_lab4p_c0_c1`: camera 0 + camera 1;
- `data/canh/epfl_lab4p_c2_c3`: camera 2 + camera 3.

Kèm theo là file hiệu chỉnh do EPFL đo (`calibration-4p.txt`). Với mỗi camera có:
- ma trận **G_sàn**, đổi điểm trong ảnh thành điểm trên **bản đồ sàn nhìn từ trên xuống** ("top view");
- với camera 0, 1: ma trận **G_đầu** cho mặt phẳng ngang cao 1,75 m, tức ngang đỉnh đầu người;
- với camera 2, 3: độ cao của đường ngang chứa mọi đỉnh đầu.

## Chưa có đáp án thì kiểm tra thế nào

Bộ video này **chưa có đáp án** (khung người đúng ở từng frame), nên chưa tính được MOTA, IDF1 hay số người
ghép đúng giữa hai camera. Trong lúc chờ, hệ thống được kiểm tra bằng những thứ đo được mà không cần đáp án:

| Kiểm tra | Cách đo | Kết quả trên camera 0 |
|---|---|---|
| Khung có đúng cỡ người không | Từ chân khung, dùng hiệu chỉnh EPFL tính chiều cao người mong đợi: đầu = G_đầu⁻¹ · G_sàn · chân. So với chiều cao khung | Trung vị 0,90 lần chiều cao mong đợi |
| Có tìm thấy người không | Tỉ lệ frame có ít nhất một khung (frame 900–1300) | 40/41 frame |
| Có nhiều tên thừa không | Số ID và track dài nhất trong 1000 frame đầu (tối đa 4 người) | 7 ID, có người giữ tên liền 844 frame |
| Màu áo phân biệt người không | Độ giống màu của cùng một người cách vài frame, so với hai người khác nhau | Cùng người 0,82 và 0,86; khác người 0,64 và 0,48 |

Các con số này nằm trong test tự động (`tests/test_that_*.py`, `tests/test_reid.py`). Khung người dùng trong test
được chọn bằng mắt trên frame thật (`tests/du_lieu_that.py`).

Khi cải thiện hệ thống, mỗi lần chạy được chấm bằng `python -m tools.cham_khong_dap_an` (hoặc `tools/hieu_chinh.py`):
- **từng camera (QĐ-34):** số tên, độ phủ người, track ma, cú nhảy;
- **ghép hai camera (QĐ-35):** chân của hai track được gán cùng tên, đổi ra bản đồ sàn bằng hiệu chỉnh EPFL, phải
  cách nhau không quá khoảng 0,75 m. Nhờ vậy biết được cặp nào **lệch**, tức chắc chắn ghép nhầm. Kiểm tra này chỉ
  khách quan với biến thể chỉ dùng ngoại hình, vì biến thể kia đã dùng chính vị trí sàn để ghép.

## Muốn có điểm thật: nhóm tự khoanh đáp án

```bash
python tools/annotate.py  --video data/canh/epfl_lab4p_c0_c1/cam_a/video.avi --out data/canh/epfl_lab4p_c0_c1/cam_a/gt_tracking.csv
python tools/verify_gt.py --video data/canh/epfl_lab4p_c0_c1/cam_a/video.avi --gt data/canh/epfl_lab4p_c0_c1/cam_a/gt_tracking.csv --out outputs/gt_check.mp4
```

- Khoanh người ở các **frame mốc** (5 frame một lần). Giữa hai mốc, công cụ tự nội suy, nhưng chỉ frame mốc được chấm.
- Làm cho cả hai camera. **Cùng một người thì dùng cùng một số ID ở hai camera.**
- Viết thêm `gt_reid.csv` (`person,cam_a_id,cam_b_id`).
- Đủ đáp án thì `run.py` và `tools/hieu_chinh.py` tự in MOTA, IDF1 và số người ghép đúng. Lúc đó mới hiệu chỉnh
  tham số theo điểm thật ([14](14_hieu_chinh_tham_so.md)).

Chỉ cần khoanh khoảng 1–2 phút video, tức 300–600 frame mốc mỗi camera, là đủ để so sánh các cách làm.

## Phép chuyển điểm sàn giữa hai camera (tools/epfl_homography.py)

`H_AB = G_B⁻¹ · G_A`: điểm chân ở ảnh A đổi ra bản đồ sàn bằng G_A, rồi từ bản đồ sàn vào ảnh B bằng G_B⁻¹. File
`homography.json` của hai cảnh đã được tạo sẵn từ `calibration-4p.txt`.

## Thầy có thể hỏi

- *"Video lấy ở đâu?"* Bộ "Multi-camera pedestrians video" của EPFL (phòng thí nghiệm, 4 người, 4 camera), nhóm
  đưa vào dự án. Chỉ dùng cho nghiên cứu; báo cáo trích dẫn bài báo của EPFL (`data/goc/epfl_lab4p/nguon.txt`).
- *"Chưa có đáp án thì sao biết hệ thống đúng?"* Hiện chỉ kiểm tra được những thứ không cần đáp án: khung đúng cỡ
  người theo hiệu chỉnh camera, số tên thừa, màu áo, và vị trí sàn của hai track được ghép giữa hai camera. Muốn có MOTA, IDF1 thì nhóm phải khoanh đáp án bằng
  `tools/annotate.py`. Đó là việc của người, không để máy làm.
