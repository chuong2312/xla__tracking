# 14. Hiệu chỉnh tham số: những chỗ nhóm tự chỉnh

> Mọi tham số nằm trong `config.yaml`. Công cụ chấm điểm: `tools/hieu_chinh.py`.
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).
> Đây là **hiệu chỉnh tham số** (dò lưới bằng tay), **không phải huấn luyện**: máy không học gì, nhóm chạy thử vài giá trị rồi chọn giá trị tốt nhất.

## 1. Chấm điểm sau khi sửa config

```bash
python -m tools.hieu_chinh                    # chấm config.yaml hiện tại (~3 phút)
python -m tools.hieu_chinh --no-real          # bỏ video thật, nhanh hơn (~1 phút)
python -m tools.hieu_chinh --set "people_tracker.max_misses=[5, 10, 20]"     # thử nhiều giá trị một lúc
python -m tools.hieu_chinh --set "detection.min_area=[150, 300]" --set "people_tracker.iou_gate=[0.1, 0.2]"
```

- Mỗi `--set` được thử **riêng**: chỉ đổi đúng tham số đó, các tham số khác giữ như `config.yaml`. Luôn có thêm lượt `hien_tai` để so.
- Kết quả in ra màn hình và ghi vào `outputs/hieu_chinh/<thời gian>/bang_diem.csv`. Config của từng lượt nằm trong thư mục con.

**Đọc bảng điểm:**

| Cột | Nghĩa | Tốt khi |
|---|---|---|
| `MOTA` | Trung bình 10 camera của 5 cảnh tổng hợp: khoanh đúng người được bao nhiêu (trừ đi bỏ sót, báo nhầm, đổi tên) | cao |
| `IDF1` | Giữ đúng tên được bao lâu | cao |
| `DIEM` | (MOTA + IDF1) / 2, **con số chính để so** | cao |
| `nhan_dang_lai` | Số người ghép đúng giữa 2 camera trên 20 người (5 cảnh × 4): chỉ ngoại hình \| ngoại hình + sàn | cao |
| `epfl…/cam_a`, `cam_b` | Video thật, chưa có đáp án: số ID và số track mỗi frame | ít ID; số track mỗi frame **không tụt mạnh** (tụt = đang bỏ sót người) |

## 2. Quy trình nên làm

1. Đổi **một** tham số (dùng `--set` với 2–3 giá trị quanh giá trị hiện tại).
2. Giữ giá trị mới khi `DIEM` tăng **và** video thật không xấu đi (số ID không tăng nhiều, track mỗi frame không tụt).
3. Sửa giá trị đó trong `config.yaml`, chạy lại `python -m tools.hieu_chinh` để có mốc mới.
4. Gộp nhiều thay đổi tốt thì **phải chấm lại bản gộp**: các tham số ảnh hưởng lẫn nhau, hai thay đổi riêng lẻ đều tốt có khi gộp lại thì tệ.
5. Ghi lại vào bảng ở mục 4 (giá trị đã thử, điểm) để báo cáo.

**Cẩn thận "học tủ":** video tổng hợp do máy vẽ, sạch hơn thực tế. Một giá trị chỉ tốt trên tổng hợp mà làm video thật xấu đi thì đừng giữ. Thay đổi chỉ hơn 0,005 điểm thì coi như bằng nhau.

## 3. Danh sách các chỗ chỉnh

Ký hiệu: ↑ = tăng giá trị, ↓ = giảm giá trị. ★ = nên thử trước.

### Bước C: tìm người (`detection`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| ★ `mog2_learning_rate` | 0,001 | MOG2 "quên" nền cũ nhanh cỡ nào; cần khoảng 1/giá trị frame để theo kịp | Người đứng yên bị nuốt vào nền nhanh hơn; theo kịp ánh sáng đổi tốt hơn | Giữ người đứng yên lâu hơn; ánh sáng đổi thì báo nhầm lâu hơn, để lại "bóng ma" |
| ★ `mog2_var_threshold` | 16 | Khác nền bao nhiêu mới tính là chuyển động | Ít báo nhầm, nhưng người mặc màu gần nền bị thủng nhiều hơn | Thấy người rõ hơn, nhưng báo nhầm nhiều hơn |
| ★ `min_area` | 150 | Khung nhỏ hơn bấy nhiêu pixel thì bỏ | Bớt khung rác và mảnh vụn; người ở rất xa có thể bị bỏ | Giữ được người ở xa, nhưng thêm rác |
| `max_area`, `min_ratio`, `max_ratio` | 40000; 1,0; 6,0 | Khung to quá, hoặc dẹt/cao bất thường thì bỏ | | |
| `close_kernel` | [3, 9] | "Cây cọ" dựng đứng để nối thân với chân | Nối được chỗ đứt lớn hơn, nhưng dễ dính hai người | Ít dính người, nhưng người dễ vỡ mảnh |
| `group_max_height_ratio` | 1,15 | Gộp hai mảnh khi khung gộp không cao quá bấy nhiêu lần một người | Gộp được nhiều mảnh hơn, nhưng dễ gộp nhầm hai người đứng trước – sau | Gộp ít hơn, nhiều mảnh rời hơn |
| `group_max_gap_ratio` | 1,0 | Khoảng hở tối đa giữa hai mảnh (× chiều cao mảnh) | | |
| `height_model.quantile` | 90 | Lấy mức cao bao nhiêu % trong mỗi dải làm "chiều cao một người" | Đường chiều cao cao hơn | Đường thấp hơn |

### Bước D: bám người (`people_tracker`, `kalman`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| ★ `iou_gate` | 0,1 | Khung mới phải chồng lên khung cũ ít nhất bấy nhiêu mới được ghép | Ít ghép nhầm sang người khác; người đi nhanh dễ bị mất | Ghép được người đi nhanh, nhưng dễ ghép nhầm |
| ★ `max_misses` | 10 | Mất dấu quá bấy nhiêu frame thì bỏ track | Giữ track lâu hơn khi bị che; nhiều track "ma" trôi lâu hơn | Bỏ track ma sớm; người bị che lâu thì đổi tên |
| ★ `merge_max_frames` | 90 | Hai người dính nhau tối đa bấy nhiêu frame thì bỏ track đứng sau | Giữ tên qua lần dính lâu (IDF1); nhưng box sai lâu hơn (MOTA) | MOTA tăng nhưng IDF1 giảm |
| `confirm_hits` | 3 | Phải thấy liên tiếp bấy nhiêu frame mới tính là người thật | Ít track rác, nhưng nhận người chậm hơn | Nhận nhanh, thêm rác |
| `merge_coverage`, `merge_keep_coverage` | 0,6; 0,3 | Ngưỡng vào / ở lại trạng thái "đang dính" (độ trễ, QĐ-24) | | |
| `recover_min_color_similarity` | 0,5 | Lúc tách nhóm, màu áo phải giống ít nhất bấy nhiêu mới nhận lại tên | Ít tráo tên khác màu; ánh sáng đổi thì mất tên | |
| `new_track_min_height_ratio` | 0,6 | Khung phải cao ít nhất bấy nhiêu lần một người mới được đặt tên mới (QĐ-27) | Ít track mảnh, nhưng người bị che một phần chậm có tên | 0 = tắt |
| `new_track_max_iou` | 0,3 | Khung thừa chồng lên track đang có từ mức này thì không tạo tên mới | | |
| `static_max_frames` | 300 | Đứng yên tối đa bấy nhiêu frame thì bỏ track | | |
| `kalman.acceleration_std_px` | 0,1 | Người đổi tốc độ, đổi hướng nhiều cỡ nào | Kalman bám nhanh theo khung mới; đoán khi bị che kém hơn | Đoán mượt khi bị che; chậm theo khi người đổi hướng |

### Nối các đoạn track bị đứt (`tracklet_linking`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| ★ `min_color_similarity` | 0,5 | Hai đoạn phải giống màu ít nhất bấy nhiêu mới nối | Ít nối nhầm, nhưng nối được ít | Nối được nhiều, có thể nối nhầm |
| `max_gap_frames` | 120 | Khoảng hở thời gian tối đa giữa hai đoạn | | |
| `max_dist_ratio` | 3,0 | Chỗ bắt đầu đoạn sau lệch khỏi chỗ đoán tối đa (× đường chéo khung) | | |

### Bước F, G: ngoại hình và ghép hai camera (`features`, `reid`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| `reid.match_threshold` | 0,3 | Điểm giống dưới mức này thì "không khớp ai" | Chắc chắn hơn, nhưng nhiều người không được ghép | Ghép được nhiều, dễ ghép nhầm |
| `reid.weight_feature`, `weight_color`, `weight_geometry` | 0,4; 0,4; 0,2 | Trọng số điểm keypoint, màu, vị trí sàn | | |
| `reid.geo_sigma_px` | 25 | Lệch bao nhiêu pixel thì điểm vị trí sàn còn khoảng 0,6 | Dễ dãi với sai số sàn | Khắt khe |
| `features.method` | sift | sift, orb, sift_auto, asift (QĐ-14) | | |
| `features.max_keypoints` | 200 | Số điểm đặc trưng tối đa mỗi ảnh | | |

### Không nên đụng

Các công tắc sau chỉ để làm thí nghiệm so sánh; bật lên thì kết quả xấu đi (đã thử, xem [11](11_tracker_moi.md), QĐ-24, QĐ-26):
- `appearance_weight` (0), `appearance_update_rate` (1,0), `template_search` (false);
- `fragment_max_vertical_overlap` (−1), `guard_shrink` (true).

## 4. Kết quả vòng dò 1 (2026-11-22)

Config gốc lúc đó như `config.yaml` hiện tại. Mỗi dòng chỉ đổi một tham số. Kết quả đầy đủ ở `outputs/hieu_chinh/01_vong1/bang_diem.csv`.

| Lượt | DIEM | MOTA | IDF1 | Nhận dạng lại | EPFL ID A / B | EPFL track/frame A / B |
|---|---|---|---|---|---|---|
| **hiện tại** | **0,796** | 0,755 | 0,836 | 18/20 \| 18/20 | 54 / 48 | 3,70 / 3,53 |
| ★ `iou_gate` = 0,2 | **0,807** | 0,755 | **0,858** | **20/20 \| 20/20** | 59 / 50 | 3,64 / 3,36 |
| ★ `max_misses` = 5 | **0,807** | **0,771** | 0,843 | 18/20 \| 18/20 | 58 / 50 | 3,52 / 3,39 |
| ★ `min_area` = 300 | 0,802 | 0,760 | 0,844 | 18/20 \| 18/20 | 51 / **38** | 3,66 / 3,41 |
| `mog2_var_threshold` = 25 | 0,802 | 0,762 | 0,842 | 16/20 \| 18/20 | 53 / **38** | 3,56 / 3,45 |
| ★ `tracklet_linking.min_color_similarity` = 0,3 | 0,800 | 0,755 | 0,844 | 18/20 \| 18/20 | 50 / 46 | 3,70 / 3,53 |
| `group_max_height_ratio` = 1,05 | 0,797 | 0,756 | 0,837 | 18/20 \| 18/20 | 54 / 60 | 3,70 / 3,76 |
| `close_kernel` = [3, 5] | 0,795 | 0,752 | 0,838 | 18/20 \| 18/20 | 58 / **34** | 3,73 / 3,30 |
| `confirm_hits` = 2 | 0,796 | 0,755 | 0,836 | 18/20 \| 18/20 | 53 / 54 | 3,72 / 3,67 |
| `confirm_hits` = 5 | 0,794 | 0,753 | 0,835 | 16/20 \| 18/20 | 50 / 35 | 3,64 / 3,36 |
| `new_track_max_iou` = 0,1 | 0,795 | 0,754 | 0,836 | 18/20 \| 18/20 | 50 / 49 | 3,66 / 3,52 |
| `max_gap_frames` = 60 / 250; `max_dist_ratio` = 2 / 5; `group_max_gap_ratio` = 0,5 / 1,5; `min_area` = 80; `new_track_max_iou` = 0,5 | 0,796 | 0,755 | 0,836 | 18/20 \| 18/20 | 51–60 / 46–60 | gần như không đổi |
| `reid.match_threshold` = 0,2 / 0,4 | 0,796 | 0,755 | 0,836 | 18/20 \| 18/20 | 54 / 48 | không đổi |
| `mog2_var_threshold` = 10 | 0,790 ↓ | 0,748 | 0,833 | 16/20 \| 18/20 | 59 / 63 | 3,62 / 3,61 |
| `iou_gate` = 0,05 | 0,790 ↓ | 0,754 | 0,825 | 16/20 \| 16/20 | 43 / 46 | 3,72 / 3,47 |
| `merge_max_frames` = 45 | 0,788 ↓ | **0,776** | 0,801 | 17/20 \| 17/20 | 55 / 46 | 3,59 / 3,40 |
| `merge_max_frames` = 150 | 0,784 ↓ | 0,741 | 0,827 | 19/20 \| 19/20 | 54 / 48 | 3,70 / 3,53 |
| `min_color_similarity` = 0,7 | 0,782 ↓ | 0,755 | 0,809 | 15/20 \| 15/20 | 62 / 60 | 3,70 / 3,53 |
| `group_max_height_ratio` = 1,3 | 0,781 ↓ | 0,743 | 0,819 | 15/20 \| 17/20 | 54 / 45 | 3,70 / 3,54 |
| `max_misses` = 20 | 0,776 ↓ | 0,727 | 0,824 | 18/20 \| 18/20 | 55 / 52 | 3,94 / 3,77 |
| `close_kernel` = [5, 15] | 0,768 ↓ | 0,715 | 0,820 | 18/20 \| 18/20 | 42 / 45 | 3,37 / 3,47 |

**Đọc kết quả:**
- Đáng thử nhất: `iou_gate` 0,2 (nhận dạng lại 20/20, IDF1 cao nhất) và `max_misses` 5 (MOTA cao nhất). Trên EPFL hai giá trị này làm số ID tăng nhẹ, nên cần xem thêm video debug.
- `min_area` 300 và `tracklet_linking.min_color_similarity` 0,3 tốt ở cả tổng hợp lẫn EPFL.
- `mog2_var_threshold` 25 tốt cho MOTA/IDF1 nhưng nhận dạng lại giảm (16/20): đánh đổi.
- `merge_max_frames` là đánh đổi rõ: giảm thì MOTA tăng nhưng IDF1 giảm.
- Nhóm nối tracklet (`max_gap_frames`, `max_dist_ratio`) và `reid.match_threshold` gần như không ảnh hưởng ở các giá trị đã thử.

## 5. Bản gộp gợi ý (chưa áp dụng vào `config.yaml`)

File `outputs/hieu_chinh/02_goi_y_gop/config_goi_y.yaml` = config hiện tại + `iou_gate` 0,2, `max_misses` 5, `min_area` 300, `tracklet_linking.min_color_similarity` 0,3.

| Lượt | DIEM | MOTA | IDF1 | Nhận dạng lại | EPFL ID A / B | EPFL track/frame A / B |
|---|---|---|---|---|---|---|
| `config.yaml` hiện tại | 0,796 | 0,755 | 0,836 | 18/20 \| 18/20 | 54 / 48 | 3,70 / 3,53 |
| **Gộp cả 4 thay đổi** | **0,820** | 0,772 | **0,867** | **20/20 \| 20/20** | 51 / 45 | 3,44 / 3,33 |
| Gộp 3 thay đổi (giữ `iou_gate` 0,1) | 0,814 | **0,776** | 0,853 | 18/20 \| 18/20 | **44 / 38** | 3,48 / 3,28 |

Bản gộp 4 thay đổi tốt hơn `config.yaml` hiện tại ở mọi cột của video tổng hợp, và ít ID hơn trên EPFL. Track mỗi frame trên EPFL giảm nhẹ (3,70 → 3,44), nên xem video debug để chắc không bỏ sót thêm người.

Muốn dùng thì chép 4 giá trị trên vào `config.yaml`, hoặc chạy thử trước:

```bash
python run.py --scene data/real/epfl_4p_c0_c1 --debug-video --config outputs/hieu_chinh/02_goi_y_gop/config_goi_y.yaml --out outputs/he_thong/epfl_goi_y
```
