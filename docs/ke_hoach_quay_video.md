# Kế hoạch quay video

> **Mục tiêu:** quay dữ liệu đúng giả định "camera cố định" để trừ nền hoạt động được, đủ kịch bản cho mọi phần đánh giá, và có ground truth cho dev và test tách biệt.
> **Căn cứ:** XLA 2.11, 2.12, 2.15, 3.1, 3.3, 3.5, 3.6, 3.7.
> **Video thật lấy trên mạng để thử thêm:** xem [tim_video_thuc_te.md](tim_video_thuc_te.md).
> **Lý do phải quay lại:** video S1 hiện tại được quay cầm tay. Camera trôi khoảng 25 px trong 15 giây, nên trừ nền bắt toàn bộ các cạnh tường (xem QĐ-01 trong [quyet_dinh_thiet_ke.md](quyet_dinh_thiet_ke.md)).

## 1. Chuẩn bị

- 2 điện thoại và **2 chân máy**. Nếu không có chân máy thì kê thật chắc và không chạm vào máy trong lúc quay.
- Băng keo màu để đánh dấu **4–6 điểm trên sàn**.
- Một tấm thẻ xám hoặc tờ giấy trắng, dùng để cân bằng màu giữa hai camera.
- Pin hoặc sạc dự phòng, và bộ nhớ còn trống.
- 4 người tham gia.
  - Các kịch bản S1–S3: mỗi người giữ cố định một bộ quần áo.
  - Kịch bản S4: cần **2 người mặc áo cùng tông màu**.

## 2. Cài đặt điện thoại

Cài cả hai máy giống nhau:
- Quay **ngang**, 1080p, 30 fps. Nếu máy cho chọn fps cố định thì chọn.
- **Khóa phơi sáng và lấy nét** (chạm giữ lên màn hình để bật AE/AF lock). Nhờ vậy độ sáng không tự đổi khi có người đi qua.
- Tắt HDR tự động, tắt chế độ làm đẹp, không zoom.
- Tắt chống rung điện tử nếu máy cho tắt. Khi đã có chân máy, chống rung có thể làm khung hình trôi nhẹ.

## 3. Bố trí camera (chồng vùng nhìn)

- Dùng **2 camera** (thầy cho tối đa 4). Hai camera nhìn **cùng một khu vực** từ hai góc lệch nhau khoảng 45–90°.
- Đặt máy cao khoảng 1,8–2,5 m và chúc xuống chéo, để thấy **cả người lẫn sàn**.
- Khi đứng giữa khu vực, người nên cao **ít nhất khoảng 1/3 khung hình**.
- Cả hai camera phải cùng thấy **ít nhất 4 điểm băng keo**. Các điểm không thẳng hàng và rải rộng khắp vùng đi lại, để tính được homography mặt sàn.
- Tránh những thứ sau trong nền:
  - Kính hoặc gương, vì có phản chiếu.
  - Nắng chiếu thẳng hoặc đèn nhấp nháy.
  - Cửa đóng mở liên tục (trừ cảnh cửa).
- Với cảnh cửa: ít nhất một camera phải thấy rõ cửa và lối đi qua cửa.
- Chụp lại ảnh vị trí chân máy để quay lại đúng như cũ khi cần.

## 3b. Bố trí riêng cho cảnh cửa: hành lang + cửa (QĐ-17)

Cảnh cửa không dùng kiểu "hai camera cùng nhìn một khu vực", mà dùng bố trí giống camera an ninh thật:

```
             tường đối diện cửa              [Cam B] nhìn thẳng vào cửa
                                                │
 [Cam A] ●  →  →  người đi dọc hành lang  →  ↘  ▼
 (đầu/góc hành lang)                          ↘
 tường có cửa ──────────────────────────────[ CỬA ]──────
                                               phòng
```

- **Camera A** đặt ở đầu hoặc góc hành lang, nhìn dọc hành lang: thấy người đi tới rồi rẽ vào cửa.
- **Camera B** gắn ở tường đối diện cửa, nhìn thẳng vào cửa: phải thấy **trọn lối cửa và chân người** khi đi qua. Đếm cửa chủ yếu dựa vào camera này.
- **Vỗ tay ngay ở cửa** lúc bắt đầu, chỗ cả hai camera cùng thấy. Với bố trí này, đồng bộ giờ là **bắt buộc**, vì hai camera được ghép bằng thời gian qua cửa ([07_ghep_su_kien_cua.md](07_ghep_su_kien_cua.md)).
- Dán **ít nhất 4 điểm băng keo** ở vùng sàn quanh cửa mà cả hai camera cùng thấy, để dùng homography.

Mỗi lượt quay dài 30–40 giây, gồm đủ các tình huống sau:

| Tình huống | Thử điều gì |
|---|---|
| 1 người đi dọc hành lang rồi rẽ vào cửa | Trường hợp cơ bản |
| 2 người vào **lần lượt**, cách nhau 1–2 giây | Ghép đúng thứ tự theo thời gian |
| 2 người qua cửa **gần như cùng lúc** (cùng chiều, hoặc một vào một ra) | Trường hợp khó; phải nhờ ngoại hình |
| 1 người **đi ngang qua cửa mà không vào** | Không được đếm nhầm |
| 1 người từ trong phòng **đi ra** | Đếm RA |
| 1 người **dừng lại ở cửa** vài giây rồi mới vào | Độ lệch thời gian giữa hai camera lớn hơn |

## 4. Quy trình mỗi lượt quay

1. Bấm quay cả hai máy.
2. Quay **3–5 giây cảnh trống**, không có ai trong khung. Đoạn này dùng để khởi tạo ảnh nền.
3. Một người giơ **thẻ xám** ở giữa khu vực trong khoảng 2 giây.
4. **Vỗ tay một cái thật rõ** ở chỗ cả hai camera cùng thấy. Tiếng và hình vỗ tay dùng để đồng bộ hai video.
5. Diễn kịch bản trong 20–40 giây.
6. Quay thêm 2 giây cảnh trống, rồi dừng cả hai máy.
7. Ghi vào sổ: tên file trên mỗi máy, kịch bản nào, lượt dev hay test.

## 5. Danh sách kịch bản

S1–S4 lấy theo roadmap gốc. Nếu roadmap gốc mô tả khác thì sửa bảng này cho khớp.

| Kịch bản | Nội dung | Kiểm tra phần nào | Số lượt |
|---|---|---|---|
| S1 | 4 người đi qua lại thưa, hạn chế che nhau | C, D, G cơ bản | 2 (dev, test) |
| S2 | Đi giao cắt và che khuất nhau | D (blob gộp và tách) | 2 |
| S3 | Một người **đứng yên ít nhất 5 giây** giữa khung | C, D (trạng thái `static`) | 2 |
| S4 | Trong 4 người có **2 người mặc áo cùng tông màu**; hai người này đi giao cắt nhau | D, G (không dựa vào màu) | 2 |
| Cửa (**bắt buộc**) | Quay theo **bố trí hành lang + cửa** (mục 3b), gồm đủ các tình huống liệt kê ở đó | E (thầy yêu cầu), ghép hai camera qua cửa | 2 |
| S5 *(tùy chọn)* | Như S1 nhưng hai camera lệch góc rất lớn | Ablation góc camera (XLA 3.5) | 1–2 |

Tối thiểu cần 5 kịch bản × 2 lượt × 2 camera = **20 video**.

Lưu ý cho báo cáo: dev và test dùng **cùng 4 người**, nên kết quả Re-ID sau khi chỉnh trọng số trên dev có thể lạc quan (XLA 3.1).

## 6. Sau khi quay

1. **Chuyển sang frame rate cố định.** Điện thoại hay quay frame rate thay đổi (VFR), làm hai video lệch dần theo thời gian (XLA 3.6). Giữ lại âm thanh để còn tìm tiếng vỗ tay:
   ```bash
   ffmpeg -i raw.mp4 -vf fps=30 -c:v libx264 -crf 18 -c:a aac video.mp4
   ```
2. **Xếp file vào thư mục:**
   ```
   data/real/<dev|test>/<S1|S2|S3|S4|door|S5>/<cam_a|cam_b>/video.mp4
   ```
   Giữ bản gốc ở chỗ khác, không đưa vào repo.
3. **Đồng bộ.** Tìm frame có tiếng vỗ tay ở mỗi camera. Độ lệch giữa hai camera ghi vào `cameras.cam_b.frame_offset`.
4. **Homography mặt sàn.** Click các điểm băng keo theo cùng một thứ tự ở hai camera, bằng công cụ `tools/calibrate_floor.py` (sẽ viết). Sau đó tính `cv2.findHomography` và lưu kết quả vào config.

## 7. Quy ước gán nhãn (XLA 3.3)

- Gán nhãn trên khung hình **đã resize về 640×360**, và ghi rõ điều này trong báo cáo.
- Box chỉ bao **phần nhìn thấy** của người, không đoán phần bị che.
- Thêm cột `occluded`: 0 là không bị che, 1 là che một phần, 2 là che nặng. Frame có `occluded = 2` được bỏ qua khi tính P/R của detector.
- Gán tay mỗi 5 frame (keyframe), các frame ở giữa thì nội suy. **P/R của detector chỉ tính trên keyframe gán tay.**
- File `gt_reid.csv` ghi người nào ở cam A tương ứng với người nào ở cam B.
- Tuyệt đối **không dùng công cụ tự gán nhãn** chạy bằng mô hình (XLA 1).

## 8. Ước tính công gán nhãn (XLA 3.7)

- 20 video × 30 giây × 30 fps ÷ 5 = **3.600 keyframe**.
- Với 4 người, trung bình khoảng 3,5 người mỗi frame, tức khoảng **12.600 box**.
- Mỗi box mất 3–5 giây, tổng cộng khoảng **10,5–17,5 giờ**.

Chia đều cho các thành viên và đưa vào lịch ngay từ đầu.
