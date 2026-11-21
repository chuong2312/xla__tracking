# outputs/

Mọi thứ trong thư mục này đều sinh lại được từ code và data, nên không commit (xem `.gitignore`). Riêng file này được commit.

| Đường dẫn | Nội dung |
|---|---|
| `detections.csv`, `detector_debug.mp4`, `tracks.csv`, `tracking_debug.mp4`, `gt_check.mp4` | Output mặc định của các lệnh trong README gốc. Mỗi lần chạy lại sẽ ghi đè. |
| `explain/` | Ảnh minh họa của các module mới (`video_tong_hop`, `homography_san`, `ghep_cap`, `kalman`, `dem_qua_cua`). Tạo lại bằng các lệnh trong README gốc. |
| `smoke/` | Lần chạy thử công cụ gán nhãn `tools/annotate.py`. |
| `synthetic_d2/00_initial/` | Lần chạy đầu của tracker trên video tổng hợp `cross` và `occlude`. |
| `synthetic_d2/01–03_corrective_pass*/` | Các vòng sửa tracker. Mỗi vòng gồm `tracks.csv` và video debug. |
| `synthetic_d2/04_pass4/` | Vòng đầu tiên có diagnostics JSONL. |
| `synthetic_d2/05_pass5/` | Chạy `tracker_only` (detection oracle) và `integration` (detector + tracker), có manifest. |
| `synthetic_d2/06_pass5_audit/`, `07_pass5_audit2/` | Script và bảng phân tích: ablation a–e, độ nhạy theo seed, các frame bị đổi ID. |
| `real_dev_S1/base/` | Video thật S1 cam_a chạy với config lúc đó. `diag/` chứa ảnh so sánh frame với foreground mask. |
| `real_dev_S1/lr01/` | Giống `base` nhưng `detector.learningRate: 0.01` (xem `config_lr01.yaml`). |
| `real_dev_S1/r1/` | Giống `base` nhưng `min_area: 120`, `max_area: 3000` (xem `config_r1.yaml`). |

Quy ước cho các lần chạy mới: tạo một thư mục riêng `outputs/<giai_doan>/<NN_ten_ngan>/`, đặt kèm file config đã dùng, và ưu tiên chạy bằng `tools/trace_tracking.py` để có manifest.
