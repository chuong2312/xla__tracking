# 14. Hiệu chỉnh tham số: những chỗ nhóm tự chỉnh

> Mọi tham số nằm trong `config.yaml`. Công cụ chấm điểm: `tools/hieu_chinh.py`.
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).
> Đây là **hiệu chỉnh tham số** (dò lưới bằng tay), **không phải huấn luyện**: máy không học gì, nhóm chạy thử vài giá trị rồi chọn giá trị tốt nhất.

## 1. Chấm điểm sau khi sửa config

Chỉ chấm trên **video thật** (QĐ-31): mọi cảnh trong `data/canh/` ([data/README.md](../data/README.md)).

```bash
python -m tools.hieu_chinh                    # chấm config.yaml hiện tại trên mọi cảnh (khoảng 5–10 phút)
python -m tools.hieu_chinh --chi-dap-an       # chỉ các cảnh có đáp án, nhanh hơn
python -m tools.hieu_chinh --set "people_tracker.max_misses=[2, 3, 5]"      # thử nhiều giá trị một lúc
python -m tools.hieu_chinh --set "reid.match_threshold=[0.1, 0.2]" --set "tracklet_linking.min_color_similarity=[0.5, 0.6]"
```

- Mỗi `--set` được thử **riêng**: chỉ đổi đúng tham số đó, các tham số khác giữ như `config.yaml`. Luôn có thêm lượt `hien_tai` để so.
- Kết quả in ra màn hình và ghi vào `outputs/hieu_chinh/<thời gian>/bang_diem.csv`. Config của từng lượt nằm trong thư mục con.

**Đọc bảng điểm:**

| Cột | Nghĩa | Tốt khi |
|---|---|---|
| `MOTA` | Trung bình các camera của **cảnh có đáp án** (nhóm khoanh bằng `tools/annotate.py`; hiện chưa có): khoanh đúng người được bao nhiêu, đã trừ bỏ sót, báo nhầm và đổi tên | cao |
| `IDF1` | Giữ đúng tên được bao lâu | cao |
| `DIEM` | (MOTA + IDF1) / 2, **con số chính để so** | cao |
| `nhan_dang_lai` | Số người ghép đúng giữa 2 camera, cộng mọi cảnh có đáp án: chỉ ngoại hình \| ngoại hình + sàn | cao |
| `<cảnh>/cam_a`, `cam_b` | Mọi cảnh (kể cả cảnh chưa có đáp án): số ID và số track mỗi frame | ít ID; số track mỗi frame **không tụt mạnh** (tụt là đang bỏ sót người) |

## 2. Quy trình nên làm

1. Đổi **một** tham số (dùng `--set` với 2–3 giá trị quanh giá trị hiện tại).
2. Giữ giá trị mới khi `DIEM` tăng, nhận dạng lại không giảm, và các cảnh chưa có đáp án không xấu đi (số ID không tăng nhiều, track mỗi frame không tụt).
3. Sửa giá trị đó trong `config.yaml`, chạy lại `python -m tools.hieu_chinh` để có mốc mới.
4. Gộp nhiều thay đổi tốt thì **phải chấm lại bản gộp**: các tham số ảnh hưởng lẫn nhau, hai thay đổi riêng lẻ đều tốt có khi gộp lại thì tệ.
5. Ghi giá trị đã thử và điểm vào mục 4 để báo cáo.

**Cẩn thận "học tủ":** đừng chỉnh chỉ cho một cảnh hay một cặp camera. Nên chỉnh trên một cặp camera rồi kiểm tra lại trên cặp kia. Thay đổi chỉ hơn 0,005 điểm thì coi như bằng nhau.

## 3. Danh sách các chỗ chỉnh

Ký hiệu: ↑ = tăng giá trị, ↓ = giảm giá trị. ★ = nên thử trước.

### Ổn định hình (`stabilization`)

| Tham số | Hiện tại | Nghĩa | ↑ thì | ↓ thì |
|---|---|---|---|---|
| `enabled` | true | Bật bù camera trôi/rung | | false: tắt (chỉ nên tắt khi chắc camera đứng yên) |
| `min_shift_px` | 1,0 | Góc ảnh dời ít hơn bấy nhiêu thì coi như camera đứng yên | Bỏ qua rung nhẹ | Bù cả rung rất nhỏ (ảnh bị nội suy nhiều hơn) |
| `ransac_threshold_px` | 2,0 | Điểm lệch quá mức này bị coi là điểm trên người, không phải nền | | |
| `max_corners`, `min_inliers` | 300, 20 | Số điểm nền dùng để đo độ trôi; tối thiểu bấy nhiêu điểm khớp | | |

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
| `height_model.max_residual` | 0,15 | Đường chiều cao lệch các điểm quá mức này thì không dùng mô hình (QĐ-30) | Dùng mô hình cả khi khớp kém | Tắt mô hình sớm hơn |
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

## 4. Mốc hiện tại và lịch sử

**Mốc trên video của nhóm** (2026-11-23, EPFL 4 người, chưa có đáp án): số ID sau khi nối 42 / 26 (c0 + c1) và
56 / 75 (c2 + c3); chi tiết ở [12](12_he_thong.md). Vì **chưa có cảnh nào có đáp án**, cột `MOTA`, `IDF1`, `DIEM`
trong bảng điểm còn trống. Khi chưa có đáp án, chỉ nên chỉnh những thay đổi làm giảm số ID mà không làm tụt số
track mỗi frame, và phải xem lại video debug.

**Lịch sử:** các vòng dò ngày 2026-11-22 (QĐ-29) chấm chủ yếu trên video máy vẽ, nên **không còn dùng làm căn cứ**
(QĐ-31). Các giá trị đang dùng trong `config.yaml` là kết quả của các vòng đó, nên **cần dò lại khi có đáp án**.
Nên dò trước các tham số sau:
- `reid.match_threshold` (0,3): trên video thật, điểm ghép thường thấp hơn trên video máy vẽ;
- `tracklet_linking.min_color_similarity` (0,3): đo trên video của nhóm, cùng người giống màu 0,82–0,86, khác người 0,48–0,64, nên 0,3 có thể quá dễ dãi;
- `people_tracker.recover_min_color_similarity` (0,5): cùng lý do;
- `reid.weight_feature` / `weight_color` / `weight_geometry` (0,4 / 0,4 / 0,2): ở ảnh 360×288 điểm đặc trưng khớp ít, nên có thể cần tăng trọng số vị trí sàn.
