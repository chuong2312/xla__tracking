# XLA Tracking

## Cài đặt

Yêu cầu Python 3.10 trở lên. Tạo môi trường ảo rồi cài dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy lệnh từ thư mục `xla_tracking`.

## Phương án mới và các module đã làm

Phương án mới nằm trong `docs/de_xuat_phuong_an.md`. Gặp từ khó thì tra `docs/tu_dien_thuat_ngu.md`. Các module dưới đây đã làm xong; mỗi module có một tài liệu giải thích bằng tiếng Việt và một lệnh xuất ảnh minh họa:

| Module | Tài liệu | Lệnh minh họa |
|---|---|---|
| Video tổng hợp v2 | `docs/01_video_tong_hop.md` | `python -m tools.make_synthetic_v2 --out data/synthetic_v2 --explain outputs/explain/video_tong_hop` |
| Homography mặt sàn | `docs/02_homography_san.md` | `python tools/calibrate_floor.py --cam-a … --cam-b … --out … --explain …` |
| Ghép cặp Hungarian | `docs/03_ghep_cap_hungarian.md` | `python -m tools.explain_assignment` |
| Kalman | `docs/04_kalman.md` | `python -m tools.explain_kalman` |
| Đếm qua cửa | `docs/05_dem_qua_cua.md` | `python -m tools.explain_door` |
| MOTA, MOTP, IDF1 | `docs/06_cham_diem_mota_idf1.md` | `python -m src.metrics track <tracks.csv> <gt.csv>` |
| Ghép lượt qua cửa giữa 2 camera | `docs/07_ghep_su_kien_cua.md` | `python -m tools.explain_door_matching` |
| Đặc trưng ngoại hình (SIFT/ORB/ASIFT, màu) | `docs/08_dac_trung_ngoai_hinh.md` | xem module 09 |
| Nhận dạng lại giữa 2 camera | `docs/09_nhan_dang_lai.md` | `python -m tools.run_reid --scene data/synthetic_v2/s4` và `python -m tools.explain_reid` |

**Chạy cả hệ thống** (tìm người → bám người → đếm cửa → nhận dạng lại giữa 2 camera → chấm điểm):

```bash
python run.py --scene data/synthetic_v2/hanh_lang_cua --debug-video
```

Giải thích chi tiết ở `docs/10_tim_nguoi.md`, `docs/11_tracker_moi.md` và `docs/12_he_thong.md`.

Chạy test bằng `python -m pytest`. Phải có `python -m`: lệnh `pytest` trơn không tìm thấy thư mục `src`.

## Gán nhãn GT

Chạy từ thư mục `xla_tracking`: `python tools/annotate.py --video data/input.mp4 --out data/gt_tracking.csv`. Các thao tác:
- Kéo chuột để tạo box, rồi nhấn phím số 0–9 để gán ID.
- `o` đổi mức che của box vừa gán (0 không che, 1 che một phần, 2 che nặng).
- `n`/`p` chuyển keyframe sau/trước.
- `u` hoàn tác box đang chờ.
- `s` lưu; `q` lưu rồi thoát.

Thêm `--size 640 360` nếu video không theo kích thước trong `config.yaml`. Box chỉ được nội suy giữa các keyframe đã gán của cùng một ID.

File đáp án có các cột `frame,gt_id,x,y,w,h,occluded,keyframe` (QĐ-15).

Kiểm tra nhãn và xuất overlay: `python tools/verify_gt.py --video data/input.mp4 --gt data/gt_tracking.csv --out outputs/gt_check.mp4`.

## Detector MOG2

Chạy phát hiện chuyển động trên video (frame được resize một lần theo `scene.resize_width` và `scene.resize_height`):

```bash
python run_detector.py --video data/synthetic/cross/cam_a/video.mp4 --out outputs/detections.csv --debug-video outputs/detector_debug.mp4
```

Các tham số nền, morphology, diện tích và tỷ lệ hình học nằm trong section `detector` của `config.yaml`. `learningRate: 0.001` được đặt thấp để nền thích nghi chậm, giảm khả năng MOG2 hấp thụ người đứng yên; đổi lại, bóng ma chuyển động có thể tồn tại lâu hơn. HOG+SVM đã bị xóa vì là học máy (xem `docs/quyet_dinh_thiet_ke.md`, QĐ-04).

`scene.roi` là vùng giữ lại dạng `[x1, y1, x2, y2]`; `null` nghĩa là dùng toàn ảnh. `scene.ignore_regions` là danh sách vùng che dạng `[[x, y, w, h], ...]`. ROI và vùng che đều tính trên ảnh sau resize; pixel vùng che bị đưa về 0 trước connected components và trong mask trả về bởi `get_foreground_mask()`.

## Đánh giá

Metrics đọc các hợp đồng CSV hiện có và dùng Hungarian Algorithm trên chi phí `1 - IoU` để ghép box theo frame. Ngưỡng IoU mặc định `metrics.iou_threshold` là 0.5; cặp được chấp nhận khi IoU lớn hơn hoặc bằng ngưỡng. `metrics.mostly_tracked_ratio` đặt tỷ lệ frame GT tối thiểu cho chỉ số mostly tracked.

```bash
python -m src.metrics det outputs/detections.csv data/synthetic/cross/gt_tracking.csv
python -m src.metrics track outputs/tracks.csv data/synthetic/cross/gt_tracking.csv
python -m src.metrics reid outputs/global_ids.csv data/synthetic/cross/gt_reid.csv --scores outputs/reid_scores.csv --method ORB
```

Detection chỉ chấm frame có GT. Tracking chỉ xét track có `state=confirmed`. Re-ID tính từng cặp camera trong `gt_reid.csv`; thiếu global ID hoặc ID `-1` được tính là không đúng. Rank-1 chọn candidate có score cao nhất trong method chỉ định. Metrics trả số lượng và tỷ lệ; không tự tạo ground truth hay kết quả thực nghiệm.

## Tracking

Chạy MOG2 detector và Lucas-Kanade tracker trên cùng frame đã resize một lần:

```bash
python run_tracking.py --video data/synthetic/cross/cam_a/video.mp4 --out outputs/tracks.csv --debug-video outputs/tracking_debug.mp4
python -m src.metrics track outputs/tracks.csv data/synthetic/cross/gt_tracking.csv
```

Tracker dùng LK để dịch từng box theo median displacement của feature points, kiểm tra Forward-Backward, rồi gán detection bằng Hungarian và IoU gate (`matcher: greedy` bật baseline greedy). Track mới bắt đầu tentative và chỉ confirmed sau số hit liên tiếp trong `tracker.confirm_hits`; ID tăng đơn điệu, không tái sử dụng. Detection được phân loại trước association bằng IoU, coverage của từng predicted track và độ tách biệt tâm; detection merged không được dùng để cập nhật box, histogram hay tạo track. Histogram HSV chỉ lấy từ detection sạch. Track đã được xác nhận nằm trong blob được giữ tối đa `tracker.blob_grace_frames` sau lần blob cuối để chờ detection sạch; track khác vẫn theo `max_misses`. Các tham số LK, blob, histogram, matching và vòng đời nằm ở section `tracker` trong `config.yaml`.
