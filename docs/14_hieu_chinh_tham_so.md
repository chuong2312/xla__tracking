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
| ★ `min_area` | 300 | Khung nhỏ hơn bấy nhiêu pixel thì bỏ | Bớt khung rác và mảnh vụn; người ở rất xa có thể bị bỏ | Giữ được người ở xa, nhưng thêm rác |
| `max_area`, `min_ratio`, `max_ratio` | 40000; 1,0; 6,0 | Khung to quá, hoặc dẹt/cao bất thường thì bỏ | | |
| `close_kernel` | [3, 9] | "Cây cọ" dựng đứng để nối thân với chân | Nối được chỗ đứt lớn hơn, nhưng dễ dính hai người | Ít dính người, nhưng người dễ vỡ mảnh |
| `group_max_height_ratio` | 1,15 | Gộp hai mảnh khi khung gộp không cao quá bấy nhiêu lần một người | Gộp được nhiều mảnh hơn, nhưng dễ gộp nhầm hai người đứng trước – sau | Gộp ít hơn, nhiều mảnh rời hơn |
| `group_max_gap_ratio` | 1,0 | Khoảng hở tối đa giữa hai mảnh (× chiều cao mảnh) | | |
| `height_model.quantile` | 90 | Lấy mức cao bao nhiêu % trong mỗi dải làm "chiều cao một người" | Đường chiều cao cao hơn | Đường thấp hơn |

### Bước D: bám người (`people_tracker`, `kalman`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| ★ `iou_gate` | 0,2 | Khung mới phải chồng lên khung cũ ít nhất bấy nhiêu mới được ghép | Ít ghép nhầm sang người khác; người đi nhanh dễ bị mất | Ghép được người đi nhanh, nhưng dễ ghép nhầm |
| ★ `max_misses` | 3 | Mất dấu quá bấy nhiêu frame thì bỏ track | Giữ track lâu hơn khi bị che; nhiều track "ma" trôi lâu hơn | Bỏ track ma sớm; người bị che lâu thì đổi tên |
| ★ `merge_max_frames` | 90 | Hai người dính nhau tối đa bấy nhiêu frame thì bỏ track đứng sau | Giữ tên qua lần dính lâu (IDF1); nhưng box sai lâu hơn (MOTA) | MOTA tăng nhưng IDF1 giảm |
| `confirm_hits` | 3 | Phải thấy liên tiếp bấy nhiêu frame mới tính là người thật | Ít track rác, nhưng nhận người chậm hơn | Nhận nhanh, thêm rác |
| `merge_coverage`, `merge_keep_coverage` | 0,6; 0,3 | Ngưỡng vào / ở lại trạng thái "đang dính" (độ trễ, QĐ-24) | | |
| `recover_min_color_similarity` | 0,5 | Lúc tách nhóm, màu áo phải giống ít nhất bấy nhiêu mới nhận lại tên | Ít tráo tên khác màu; ánh sáng đổi thì mất tên | |
| `new_track_min_height_ratio` | 0,7 | Khung phải cao ít nhất bấy nhiêu lần một người mới được đặt tên mới (QĐ-27) | Ít track mảnh, nhưng người bị che một phần chậm có tên | 0 = tắt |
| `new_track_max_iou` | 0,3 | Khung thừa chồng lên track đang có từ mức này thì không tạo tên mới | | |
| `static_max_frames` | 300 | Đứng yên tối đa bấy nhiêu frame thì bỏ track | | |
| `kalman.acceleration_std_px` | 0,1 | Người đổi tốc độ, đổi hướng nhiều cỡ nào | Kalman bám nhanh theo khung mới; đoán khi bị che kém hơn | Đoán mượt khi bị che; chậm theo khi người đổi hướng |

### Nối các đoạn track bị đứt (`tracklet_linking`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| ★ `min_color_similarity` | 0,3 | Hai đoạn phải giống màu ít nhất bấy nhiêu mới nối | Ít nối nhầm, nhưng nối được ít | Nối được nhiều, có thể nối nhầm |
| `max_gap_frames` | 120 | Khoảng hở thời gian tối đa giữa hai đoạn | | |
| `max_dist_ratio` | 3,0 | Chỗ bắt đầu đoạn sau lệch khỏi chỗ đoán tối đa (× đường chéo khung) | | |
| `max_cost` | 1,3 | Chi phí tối đa để được nối (vị trí + màu) | Nối nhiều hơn, ít ID hơn; có thể nối nhầm | Nối ít hơn |

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

> **Giá trị trong bảng trên là giá trị đang dùng sau hai vòng hiệu chỉnh (QĐ-29).** Mục 4–6 ghi lại quá trình.

## 4. Kết quả vòng dò 1 (2026-11-22)

Config gốc lúc đó: `min_area` 150, `iou_gate` 0,1, `max_misses` 10, `new_track_min_height_ratio` 0,6, `tracklet_linking.min_color_similarity` 0,5, `max_cost` 1,0. Mỗi dòng chỉ đổi một tham số. Kết quả đầy đủ ở `outputs/hieu_chinh/01_vong1/bang_diem.csv`.

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

## 5. Bản gộp sau vòng 1 (đã áp dụng)

File `outputs/hieu_chinh/02_goi_y_gop/config_goi_y.yaml` = config hiện tại + `iou_gate` 0,2, `max_misses` 5, `min_area` 300, `tracklet_linking.min_color_similarity` 0,3.

| Lượt | DIEM | MOTA | IDF1 | Nhận dạng lại | EPFL ID A / B | EPFL track/frame A / B |
|---|---|---|---|---|---|---|
| `config.yaml` hiện tại | 0,796 | 0,755 | 0,836 | 18/20 \| 18/20 | 54 / 48 | 3,70 / 3,53 |
| **Gộp cả 4 thay đổi** | **0,820** | 0,772 | **0,867** | **20/20 \| 20/20** | 51 / 45 | 3,44 / 3,33 |
| Gộp 3 thay đổi (giữ `iou_gate` 0,1) | 0,814 | **0,776** | 0,853 | 18/20 \| 18/20 | **44 / 38** | 3,48 / 3,28 |

Bản gộp 4 thay đổi tốt hơn `config.yaml` hiện tại ở mọi cột của video tổng hợp, và ít ID hơn trên EPFL. Track mỗi frame trên EPFL giảm nhẹ (3,70 → 3,44), nên xem video debug để chắc không bỏ sót thêm người.

Bản này đã được chép vào `config.yaml` làm mốc cho vòng 2. Chạy lại riêng bản này:

```bash
python run.py --scene data/real/epfl_4p_c0_c1 --debug-video --config outputs/hieu_chinh/02_goi_y_gop/config_goi_y.yaml --out outputs/he_thong/epfl_goi_y
```

## 6. Vòng dò 2 (2026-11-22) và bản cuối

Mốc: bản gộp ở mục 5 (điểm 0,820). Kết quả đầy đủ ở `outputs/hieu_chinh/03_vong2/bang_diem.csv`.

| Lượt | DIEM | MOTA | IDF1 | Nhận dạng lại | EPFL ID A / B | EPFL track/frame A / B |
|---|---|---|---|---|---|---|
| **mốc** | **0,820** | 0,772 | 0,867 | 20/20 \| 20/20 | 51 / 45 | 3,44 / 3,33 |
| ★ `max_misses` = 3 | 0,825 | 0,779 | 0,870 | 20/20 \| 20/20 | 46 / 41 | 3,37 / 3,23 |
| ★ `tracklet_linking.max_cost` = 1,3 | 0,820 | 0,772 | 0,867 | 20/20 \| 20/20 | **36 / 38** | 3,44 / 3,33 |
| ★ `new_track_min_height_ratio` = 0,7 | 0,820 | 0,772 | 0,867 | 20/20 \| 20/20 | 41 / 40 | 3,38 / 3,24 |
| `mog2_learning_rate` = 0,0005 | 0,820 | 0,772 | 0,868 | 20/20 \| 20/20 | 39 / 42 | 3,41 / 3,35 |
| `confirm_hits` = 4 | 0,819 | 0,772 | 0,867 | 20/20 \| 20/20 | 44 / 39 | 3,43 / 3,20 |
| `iou_gate` = 0,15 / 0,3 | 0,822 / 0,825 | | | 20/20 | 46 / 40 và 56 / 32 | |
| `mog2_var_threshold` = 30 | 0,830 | 0,785 | 0,874 | **18/20** \| 20/20 | 45 / 35 | 3,24 / 3,12 |
| `merge_max_frames` = 60 | 0,825 | 0,790 | 0,861 | **18/20 \| 18/20** | 52 / 43 | 3,38 / 3,31 |
| `kalman.acceleration_std_px` = 0,2 | 0,826 | 0,779 | 0,872 | **18/20** \| 20/20 | 54 / 46 | 3,49 / 3,38 |
| `mog2_var_threshold` = 20 | 0,824 | 0,777 | 0,871 | **18/20** \| 20/20 | 52 / 33 | 3,36 / 3,23 |
| `max_misses` = 7 | 0,815 | | | 20/20 | 49 / 39 | |
| `close_kernel` = [3, 7] / [3, 12] | 0,815 / 0,808 ↓ | | | 20/20 | | |
| `merge_keep_coverage` = 0,4 | 0,809 ↓ | 0,776 | 0,842 | 16/20 | | |
| `kalman.acceleration_std_px` = 0,05 | 0,804 ↓ | 0,770 | 0,839 | 16/20 | | |
| `min_area` 200 / 450, `group_max_height_ratio` 1,1, `recover_min_color_similarity` 0,4 / 0,6, `max_size_change` 1,4 / 2,0, `tracklet_linking.min_color_similarity` 0,2 / 0,4, `new_track_min_height_ratio` 0,5 | 0,818–0,820 | | | 20/20 | gần như không đổi | |

Gộp 3 thay đổi ★ (`max_misses` 3, `max_cost` 1,3, `new_track_min_height_ratio` 0,7) rồi chấm lại (`outputs/hieu_chinh/04_gop_vong2/`):

| | DIEM | MOTA | IDF1 | Nhận dạng lại | EPFL ID A / B | EPFL track/frame A / B |
|---|---|---|---|---|---|---|
| **Bản cuối (đang dùng)** | **0,825** | 0,779 | 0,870 | **20/20 \| 20/20** | **34 / 31** | 3,29 / 3,14 |
| + `mog2_learning_rate` 0,0005 | 0,825 | 0,779 | 0,871 | 20/20 \| 20/20 | 34 / 25 | 3,30 / 3,11 |
| + `confirm_hits` 4 | 0,824 | 0,778 | 0,870 | 20/20 \| 20/20 | 34 / 23 | 3,29 / 3,03 |
| + `tracklet_linking.max_cost` 1,6 | 0,825 | 0,779 | 0,870 | 20/20 \| 20/20 | 30 / 29 | 3,29 / 3,14 |
| + `mog2_var_threshold` 30 | 0,834 | 0,791 | 0,877 | 18/20 \| 20/20 | 48 / 24 | 3,08 / 2,91 |

Ba dòng có dấu + đầu tiên không đổi điểm tổng hợp mà EPFL ít ID hơn. **Chưa áp dụng**, vì chưa có đáp án trên EPFL để biết số ID giảm là đúng hay là gộp nhầm người; có thể thử tiếp khi có đáp án. Dòng cuối làm nhận dạng lại giảm.

**Kiểm tra bản cuối trên EPFL:** tỉ lệ người nguyên vẹn được track bám giữ nguyên (89% / 92%), track "ma" giảm 12% → 7% và 13% → 8%. Ảnh debug: phần lớn người có một khung nguyên người với tên ổn định. Lỗi còn lại: người áo đen trước nền tối chỉ có khung ở đầu, 2–3 người chồng nhau bị gộp một khung.

**Điểm từng cảnh của bản cuối** (sau khi nối):

| Cảnh | MOTA A / B | IDF1 A / B | Nhận dạng lại |
|---|---|---|---|
| `s1` | 0,96 / 0,86 | 0,98 / 0,86 | 4/4 \| 4/4 |
| `s2` | 0,42 / 0,51 | 0,68 / 0,73 | 4/4 \| 4/4 |
| `s3` | 0,95 / 0,89 | 0,98 / 0,94 | 4/4 \| 4/4 |
| `s4` | 0,59 / 0,70 | 0,79 / 0,79 | 4/4 \| 4/4 |
| `s5` | 0,96 / 0,95 | 0,98 / 0,98 | 4/4 \| 4/4 |
