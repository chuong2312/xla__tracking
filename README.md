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
