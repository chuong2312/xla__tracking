# XLA Tracking

## Cài đặt

Yêu cầu Python 3.10 trở lên. Tạo môi trường ảo rồi cài dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy lệnh từ thư mục `xla_tracking`.

## Gán nhãn GT

Chạy từ thư mục `xla_tracking`: `python tools/annotate.py --video data/input.mp4 --out data/gt_tracking.csv`. Kéo chuột để tạo box, nhấn phím số 0–9 để gán ID; `n`/`p` chuyển keyframe, `u` hoàn tác box đang chờ, `e` đánh dấu ID hết từ frame hiện tại, `s` lưu và `q` lưu rồi thoát. Các box chỉ được nội suy giữa keyframe đã gán của cùng ID.

Kiểm tra nhãn và xuất overlay: `python tools/verify_gt.py --video data/input.mp4 --gt data/gt_tracking.csv --out outputs/gt_check.mp4`.

## Detector MOG2

Chạy phát hiện chuyển động trên video (frame được resize một lần theo `scene.resize_width` và `scene.resize_height`):

```bash
python run_detector.py --video data/synthetic/cross/cam_a/video.mp4 --out outputs/detections.csv --debug-video outputs/detector_debug.mp4
```

Các tham số nền, morphology, diện tích và tỷ lệ hình học nằm trong section `detector` của `config.yaml`. `learningRate: 0.001` được đặt thấp để nền thích nghi chậm, giảm khả năng MOG2 hấp thụ người đứng yên; đổi lại, bóng ma chuyển động có thể tồn tại lâu hơn. `use_hog_baseline` mặc định `false`; HOG+SVM là baseline cổ điển tùy chọn, không thay MOG2.

`scene.roi` là vùng giữ lại dạng `[x1, y1, x2, y2]`; `null` nghĩa là dùng toàn ảnh. `scene.ignore_regions` là danh sách vùng che dạng `[[x, y, w, h], ...]`. ROI và vùng che đều tính trên ảnh sau resize; pixel vùng che bị đưa về 0 trước connected components và trong mask trả về bởi `get_foreground_mask()`.
