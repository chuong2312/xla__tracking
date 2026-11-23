# Dữ liệu: chỉ video thật của nhóm

Dự án **chỉ dùng video thật** (QĐ-31) và **chỉ dùng video do nhóm đưa** (QĐ-32): không dùng ảnh hay video do
máy vẽ, không tự tải bộ dữ liệu khác về. Thư mục `data/` không đưa lên GitHub (video rất nặng), trừ file hướng
dẫn này và `video_moi/DAT_VIDEO_VAO_DAY.md`. Mỗi máy tự chép dữ liệu vào.

## Cấu trúc hiện tại

```
data/
├── video_moi/      ← THẢ VIDEO MỚI CỦA NHÓM VÀO ĐÂY (tên gì cũng được), kèm ghi_chu.txt
├── canh/           ← mỗi thư mục con là MỘT CẢNH để chạy run.py
│   ├── epfl_lab4p_c0_c1/    camera 0 + camera 1 (cam_a/video.avi, cam_b/video.avi, homography.json)
│   └── epfl_lab4p_c2_c3/    camera 2 + camera 3
└── goc/            ← file gốc, giữ nguyên để tạo lại cảnh
    └── epfl_lab4p/          4p-c0 … c3.avi (bản gốc nhóm đưa), calibration-4p.txt, nguon.txt
```

Video hiện có: bộ **EPFL phòng thí nghiệm 4 người**, 4 camera quay cùng lúc một căn phòng, mỗi video 360×288,
25 fps, 3915 frame (khoảng 2,6 phút). File `calibration-4p.txt` là ma trận hiệu chỉnh camera EPFL đo sẵn, dùng
để chuyển điểm trên sàn giữa hai camera. **Bộ này chưa có đáp án**, nên chưa chấm được MOTA, IDF1 hay số người
ghép đúng; hệ thống chỉ đếm được số tên (ID) và số track.

Một **cảnh** là một thư mục trong `data/canh/`:

| File | Bắt buộc | Ý nghĩa |
|---|---|---|
| `cam_a/video.<đuôi>` | có | Video camera A. Đuôi nào cũng được: mp4, avi, mov, mkv, m4v |
| `cam_b/video.<đuôi>` | không | Video camera B, quay **cùng lúc** với A. Không có thì hệ thống bỏ qua bước ghép người giữa hai camera |
| `cam_a/gt_tracking.csv`, `cam_b/gt_tracking.csv` | không | Đáp án khung người, để chấm MOTA, IDF1 |
| `gt_reid.csv` | không | Đáp án người nào ở A là người nào ở B, để chấm nhận dạng lại |
| `homography.json` | không | Phép chuyển điểm trên sàn từ ảnh A sang ảnh B |
| `nguon.txt` | nên có | Ai quay, ở đâu, ngày nào, giấy phép |

## Đưa video thật mới của nhóm vào

**Cách 1, dễ nhất:** chép video vào `data/video_moi/` kèm `ghi_chu.txt` (video nào là camera A, video nào là
camera B, quay ở đâu, có vỗ tay ở đầu không), rồi nhắn người làm hệ thống tạo cảnh.

**Cách 2, tự làm:**

```bash
python -m tools.tao_canh --cam-a data/video_moi/camA.mp4 --cam-b data/video_moi/camB.mp4 --out data/canh/<ten_canh>
python run.py --scene data/canh/<ten_canh> --debug-video        # kết quả ở outputs/he_thong/<ten_canh>/
```

Chỉ có một camera thì bỏ `--cam-b`. `tao_canh` **chép nguyên file**, giữ đuôi gốc, không mã hóa lại nên ảnh
không giảm chất lượng.

### Khi tự quay, cần
- **Camera đứng yên** (chân máy hoặc kẹp chắc). Bước ổn định hình chỉ bù được trôi nhẹ, không bù được lia máy.
- **Hai camera quay cùng lúc**, cùng một khu vực, hai góc khác nhau. Vỗ tay một tiếng ở đầu để căn thời gian.
- Vài giây đầu **không có người**, để hệ thống học ảnh nền.
- Đủ sáng, không quay ngược sáng; độ phân giải từ 640×360, 25–30 fps.
- Muốn dùng vị trí sàn: dán 4–6 dấu băng keo trên sàn mà cả hai camera đều thấy, rồi chạy
  `python tools/calibrate_floor.py --cam-a … --cam-b … --out data/canh/<ten_canh>/homography.json`.

## Tạo đáp án (để chấm điểm)

Muốn biết hệ thống đúng bao nhiêu, nhóm phải **tự khoanh đáp án** trên một số frame mốc:

```bash
python tools/annotate.py  --video data/canh/<ten_canh>/cam_a/video.avi --out data/canh/<ten_canh>/cam_a/gt_tracking.csv
python tools/verify_gt.py --video data/canh/<ten_canh>/cam_a/video.avi --gt data/canh/<ten_canh>/cam_a/gt_tracking.csv --out outputs/gt_check.mp4
```

- Khoanh người ở các frame mốc (mặc định 5 frame một lần), giữ **cỡ ảnh gốc** (không thêm `--size`).
- Làm cho cả `cam_a` và `cam_b`. Cùng một người thì dùng cùng một số ID ở hai camera.
- Viết `gt_reid.csv` dạng `person,cam_a_id,cam_b_id`, mỗi dòng ghi ID của cùng một người ở hai camera.
- Đủ đáp án thì `run.py` và `python -m tools.hieu_chinh` tự in MOTA, IDF1 và số người ghép đúng.

## Tạo lại hai cảnh từ file gốc

```bash
python -m tools.tao_canh --cam-a data/goc/epfl_lab4p/4p-c0.avi --cam-b data/goc/epfl_lab4p/4p-c1.avi --out data/canh/epfl_lab4p_c0_c1
python -m tools.epfl_homography --calib data/goc/epfl_lab4p/calibration-4p.txt --cam-a 0 --cam-b 1 --scene data/canh/epfl_lab4p_c0_c1
python -m tools.tao_canh --cam-a data/goc/epfl_lab4p/4p-c2.avi --cam-b data/goc/epfl_lab4p/4p-c3.avi --out data/canh/epfl_lab4p_c2_c3
python -m tools.epfl_homography --calib data/goc/epfl_lab4p/calibration-4p.txt --cam-a 2 --cam-b 3 --scene data/canh/epfl_lab4p_c2_c3
```

## Nguồn và giấy phép
Bộ video 4 người thuộc bộ dữ liệu "Multi-camera pedestrians video" của CVLab – EPFL, chỉ dùng cho nghiên cứu
(`data/goc/epfl_lab4p/nguon.txt`). Báo cáo phải trích dẫn: F. Fleuret, J. Berclaz, R. Lengagne, P. Fua,
"Multi-Camera People Tracking with a Probabilistic Occupancy Map", IEEE TPAMI 30(2):267–282, 2008.
