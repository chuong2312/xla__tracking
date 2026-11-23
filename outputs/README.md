# outputs/

Mọi thứ trong thư mục này đều sinh lại được từ code và data, nên không commit (xem `.gitignore`); riêng file này được
commit. Chỉ có kết quả trên **video thật của nhóm** (QĐ-31, QĐ-32); kết quả cũ đã dọn ngày 2026-11-23.

| Đường dẫn | Nội dung | Tạo bằng |
|---|---|---|
| `he_thong/<cảnh>/` | Kết quả cho từng cảnh: khung tìm được, track, track đã nối, ID toàn cục, `summary.json` (có điểm khi cảnh có đáp án), video debug | `python run.py --scene data/canh/<cảnh> --debug-video` |
| `hieu_chinh/<thời gian>/` | Các lượt dò tham số: mỗi lượt một thư mục con kèm `config.yaml`, bảng tổng `bang_diem.csv` | `python -m tools.hieu_chinh` |
| `reid/<cảnh>/` | Nhận dạng lại dùng khung đáp án làm track (cần cảnh có đáp án) | `python -m tools.run_reid --scene …` |
| `explain/` | Ảnh minh họa: `ghep_cap`, `kalman`, `chieu_cao`, `homography`, `nhan_dang_lai`, `on_dinh_hinh` | các lệnh `python -m tools.explain_*` |

Quy ước cho các lần chạy mới: tạo một thư mục riêng `outputs/<giai_doan>/<NN_ten_ngan>/` và đặt kèm file config đã dùng.
