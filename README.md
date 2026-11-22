# XLA Tracking

## Cài đặt

Yêu cầu Python 3.10 trở lên. Tạo môi trường ảo rồi cài dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy lệnh từ thư mục `xla_tracking`.

### Chạy trên Windows (không dùng WSL)

Dùng PowerShell, đứng ở thư mục `xla_tracking`:

```powershell
# 1. Cài Python 3.12–3.14 từ python.org (nhớ tick "Add python.exe to PATH")
py -3 -m venv .venv
.venv\Scripts\Activate.ps1        # báo lỗi quyền thì chạy trước: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
                                  # dùng cmd thay PowerShell: .venv\Scripts\activate.bat
pip install -r requirements.txt
setx PYTHONUTF8 1                 # một lần duy nhất, rồi mở lại terminal: để in tiếng Việt ra file không bị lỗi mã hóa

# 2. Dữ liệu (không có trên GitHub; chỉ video thật): chép CẢ thư mục data\ từ máy bạn cùng nhóm
#    (gồm data\canh và data\goc với 4 video của nhóm), xem data/README.md.

# 3. Kiểm tra và chạy
python -m pytest
python run.py --scene data/canh/epfl_lab4p_c0_c1 --debug-video
python -m tools.hieu_chinh
```

Trên Windows, đường dẫn trong lệnh viết bằng `/` hay `\` đều được. Điểm số có thể lệch rất ít so với máy Linux, vì bộ mã hóa video của hai hệ điều hành có thể ghi ra các frame hơi khác nhau.

## Cấu trúc thư mục

```
xla_tracking/
├── run.py              Chạy cả hệ thống trên một cảnh hai camera
├── config.yaml         Mọi tham số (chỉnh theo docs/14_hieu_chinh_tham_so.md)
├── src/                Hệ thống mới
│   ├── common/         Dùng chung: Hungarian, Kalman, homography sàn, đọc video
│   ├── detect/         Bước C, tìm người: mô hình nền, tìm box, mô hình chiều cao
│   ├── tracking/       Bước D, bám người: tracker mới, nối track bị đứt
│   ├── reid/           Bước F, G: đặc trưng ngoại hình, nhận dạng lại giữa 2 camera
│   ├── evaluation/     Chấm điểm: precision/recall, MOTA, IDF1
│   └── pipeline.py     Chạy bước C → D cho một camera
├── legacy/             Hệ thống cũ (giữ để so sánh trong báo cáo)
├── tools/              Lệnh phụ: tạo cảnh, hiệu chỉnh camera, ảnh minh họa, gán nhãn đáp án, hiệu chỉnh tham số
├── tests/              Test (tests/legacy/ là test của hệ thống cũ)
├── docs/               Tài liệu tiếng Việt, mục lục ở docs/README.md
├── data/               CHỈ VIDEO THẬT DO NHÓM ĐƯA (QĐ-31, QĐ-32), không có trên GitHub; hướng dẫn ở data/README.md
│   ├── video_moi/      ← thả video mới của nhóm vào đây
│   ├── canh/           mỗi thư mục con là một cảnh cho run.py (hiện: epfl_lab4p_c0_c1, epfl_lab4p_c2_c3, chưa có đáp án)
│   └── goc/            file gốc nhóm đưa (epfl_lab4p: 4 video avi, file hiệu chỉnh), nguồn và bản quyền (nguon.txt)
└── outputs/            Kết quả chạy (không commit), mô tả ở outputs/README.md
```

## Phương án mới và các module đã làm

Phương án mới nằm trong `docs/de_xuat_phuong_an.md`. Gặp từ khó thì tra `docs/tu_dien_thuat_ngu.md`. Các module dưới đây đã làm xong; mỗi module có một tài liệu giải thích bằng tiếng Việt và một lệnh xuất ảnh minh họa:

| Module | Tài liệu | Lệnh minh họa |
|---|---|---|
| Dữ liệu thật và đáp án | `docs/15_du_lieu_that_va_dap_an.md` | `python tools/annotate.py --video <video> --out <gt.csv>` |
| Homography mặt sàn | `docs/02_homography_san.md` | `python tools/calibrate_floor.py --cam-a … --cam-b … --out … --explain …` |
| Ghép cặp Hungarian | `docs/03_ghep_cap_hungarian.md` | `python -m tools.explain_assignment` (chạy `run.py` trước) |
| Kalman | `docs/04_kalman.md` | `python -m tools.explain_kalman` (chạy `run.py` trước) |
| MOTA, MOTP, IDF1 | `docs/06_cham_diem_mota_idf1.md` | `python -m src.evaluation.metrics track <tracks.csv> <gt.csv>` |
| Đặc trưng ngoại hình (SIFT/ORB/ASIFT, màu) | `docs/08_dac_trung_ngoai_hinh.md` | xem module 09 |
| Nhận dạng lại giữa 2 camera | `docs/09_nhan_dang_lai.md` | `python -m tools.run_reid --scene <cảnh có đáp án>` và `python -m tools.explain_reid --scene <cảnh có đáp án>` |
| Tìm người (mô hình chiều cao, gộp mảnh) | `docs/10_tim_nguoi.md` | `python -m tools.explain_height_model --video data/canh/epfl_lab4p_c0_c1/cam_b/video.mp4 --frame 1800` |
| Ổn định hình (camera trôi/rung) | `docs/10_tim_nguoi.md` | `python -m tools.explain_stabilization --video <video có camera bị trôi>` |
| Homography sàn từ hiệu chỉnh EPFL | `docs/02_homography_san.md` | `python -m tools.epfl_homography --calib data/goc/epfl_lab4p/calibration-4p.txt --cam-a 0 --cam-b 1 --scene data/canh/epfl_lab4p_c0_c1 --top-view 0 0 358 360 --explain outputs/explain/homography/epfl_lab4p_c0_c1.png` |

**Chạy cả hệ thống** (tìm người → bám người → nối track bị đứt → nhận dạng lại giữa 2 camera → chấm điểm):

```bash
python run.py --scene data/canh/epfl_lab4p_c0_c1 --debug-video          # video của nhóm, kết quả ở outputs/he_thong/
python run.py --scene data/canh/epfl_lab4p_c0_c1 --max-frames 300        # chạy thử nhanh 300 frame đầu
```

Giải thích chi tiết ở `docs/10_tim_nguoi.md`, `docs/11_tracker_moi.md`, `docs/12_he_thong.md` và `docs/13_noi_tracklet.md`.

**Chỉ dùng video thật do nhóm đưa** (QĐ-31, QĐ-32). Các cảnh có sẵn, chỗ đặt video mới và cách khoanh đáp án: `data/README.md`.
Nguồn và bản quyền: `data/goc/epfl_lab4p/nguon.txt` (báo cáo phải trích dẫn bài báo của EPFL). Video hiện **chưa có đáp án**,
nên chưa chấm được MOTA, IDF1.

**Tự chỉnh tham số:** đọc `docs/14_hieu_chinh_tham_so.md`, sửa `config.yaml`, rồi chấm lại bằng

```bash
python -m tools.hieu_chinh                                            # chấm config hiện tại
python -m tools.hieu_chinh --set "people_tracker.max_misses=[5, 10]"   # thử nhiều giá trị
```

Chạy test bằng `python -m pytest`. Phải có `python -m`: lệnh `pytest` trơn không tìm thấy thư mục `src`.

## Gán nhãn GT

Chạy từ thư mục `xla_tracking`: `python tools/annotate.py --video data/input.mp4 --out data/gt_tracking.csv`. Các thao tác:
- Kéo chuột để tạo box, rồi nhấn phím số 0–9 để gán ID.
- `o` đổi mức che của box vừa gán (0 không che, 1 che một phần, 2 che nặng).
- `n`/`p` chuyển keyframe sau/trước.
- `u` hoàn tác box đang chờ.
- `s` lưu; `q` lưu rồi thoát.

Mặc định giữ cỡ gốc của video (đúng với hệ thống mới); chỉ thêm `--size RỘNG CAO` khi thật sự cần đổi cỡ. Box chỉ được nội suy giữa các keyframe đã gán của cùng một ID.

File đáp án có các cột `frame,gt_id,x,y,w,h,occluded,keyframe` (QĐ-15).

Kiểm tra nhãn và xuất overlay: `python tools/verify_gt.py --video data/input.mp4 --gt data/gt_tracking.csv --out outputs/gt_check.mp4`.

## Detector MOG2

Chạy phát hiện chuyển động trên video (frame được resize một lần theo `scene.resize_width` và `scene.resize_height`):

```bash
python -m legacy.run_detector --video data/canh/epfl_lab4p_c0_c1/cam_a/video.avi --out outputs/detections.csv --debug-video outputs/detector_debug.mp4
```

Các tham số nền, morphology, diện tích và tỷ lệ hình học nằm trong section `detector` của `config.yaml`. `learningRate: 0.001` được đặt thấp để nền thích nghi chậm, giảm khả năng MOG2 hấp thụ người đứng yên; đổi lại, bóng ma chuyển động có thể tồn tại lâu hơn. HOG+SVM đã bị xóa vì là học máy (xem `docs/quyet_dinh_thiet_ke.md`, QĐ-04).

`scene.roi` là vùng giữ lại dạng `[x1, y1, x2, y2]`; `null` nghĩa là dùng toàn ảnh. `scene.ignore_regions` là danh sách vùng che dạng `[[x, y, w, h], ...]`. ROI và vùng che đều tính trên ảnh sau resize; pixel vùng che bị đưa về 0 trước connected components và trong mask trả về bởi `get_foreground_mask()`.

## Đánh giá

Metrics đọc các hợp đồng CSV hiện có và dùng Hungarian Algorithm trên chi phí `1 - IoU` để ghép box theo frame. Ngưỡng IoU mặc định `metrics.iou_threshold` là 0.5; cặp được chấp nhận khi IoU lớn hơn hoặc bằng ngưỡng. `metrics.mostly_tracked_ratio` đặt tỷ lệ frame GT tối thiểu cho chỉ số mostly tracked.

```bash
python -m src.evaluation.metrics det outputs/detections.csv <gt_tracking.csv của nhóm>
python -m src.evaluation.metrics track outputs/tracks.csv <gt_tracking.csv của nhóm>
python -m src.evaluation.metrics reid outputs/reid/<cảnh>/global_ids_ngoai_hinh_san.csv data/canh/<cảnh>/gt_reid.csv --scores outputs/reid/<cảnh>/reid_scores.csv --method ngoai_hinh_san
```

Detection chỉ chấm frame có GT. Tracking chỉ xét track có `state=confirmed`. Re-ID tính từng cặp camera trong `gt_reid.csv`; thiếu global ID hoặc ID `-1` được tính là không đúng. Rank-1 chọn candidate có score cao nhất trong method chỉ định. Metrics trả số lượng và tỷ lệ; không tự tạo ground truth hay kết quả thực nghiệm.

## Tracking

Chạy MOG2 detector và Lucas-Kanade tracker trên cùng frame đã resize một lần:

> Hệ thống **cũ** thu ảnh về `scene.resize_width × scene.resize_height` (640×480). Đáp án của hệ thống mới theo cỡ gốc của video (360×288), nên muốn chấm hệ thống cũ bằng đáp án đó thì chạy với một config có `scene.resize_width: 360`, `scene.resize_height: 288` (`--config`). Hệ thống mới (`run.py`) luôn làm việc trên cỡ gốc.

```bash
python -m legacy.run_tracking --video data/canh/epfl_lab4p_c0_c1/cam_a/video.avi --out outputs/tracks.csv --debug-video outputs/tracking_debug.mp4
python -m src.evaluation.metrics track outputs/tracks.csv <gt_tracking.csv của nhóm>
```

Tracker dùng LK để dịch từng box theo median displacement của feature points, kiểm tra Forward-Backward, rồi gán detection bằng Hungarian và IoU gate (`matcher: greedy` bật baseline greedy). Track mới bắt đầu tentative và chỉ confirmed sau số hit liên tiếp trong `tracker.confirm_hits`; ID tăng đơn điệu, không tái sử dụng. Detection được phân loại trước association bằng IoU, coverage của từng predicted track và độ tách biệt tâm; detection merged không được dùng để cập nhật box, histogram hay tạo track. Histogram HSV chỉ lấy từ detection sạch. Track đã được xác nhận nằm trong blob được giữ tối đa `tracker.blob_grace_frames` sau lần blob cuối để chờ detection sạch; track khác vẫn theo `max_misses`. Các tham số LK, blob, histogram, matching và vòng đời nằm ở section `tracker` trong `config.yaml`.
