# Kế hoạch quay video

> **Mục tiêu:** quay dữ liệu đúng giả định "camera cố định" để trừ nền hoạt động được, đủ kịch bản cho mọi phần đánh giá, và có ground truth cho dev và test tách biệt.
> **Căn cứ:** XLA 2.11, 2.12, 2.15, 3.1, 3.3, 3.5, 3.6, 3.7.
> **Cập nhật 2026-11-22:** chỉ quay **hành lang có nhiều người đi lại**; cảnh cửa và đếm VÀO/RA đã bỏ (QĐ-21, QĐ-22).
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

## 3. Bố trí camera: hai góc ở cùng một đầu hành lang (QĐ-22)

```
        đầu xa hành lang
   ┌───────────────────────┐
   │                       │
   │   người đi lại theo   │   hành lang rộng khoảng 3 m
   │      hai chiều        │
   │                       │
   └───────────────────────┘
 [Cam A] ●             ● [Cam B]
 (góc trái)           (góc phải)
        đầu gần (đặt camera)
```

- Dùng **2 camera** (thầy cho tối đa 4), đặt ở **cùng một đầu hành lang**: một máy ở góc trái, một máy ở góc phải, cả hai nhìn dọc vào trong.
- Đặt máy cao khoảng 1,8–2,5 m và chúc xuống, để thấy **cả người lẫn sàn** dọc hành lang.
- Người ở giữa hành lang nên cao ít nhất khoảng 1/4 khung hình. Nếu hành lang quá dài thì chỉ dùng đoạn 8–10 m gần camera.
- Cả hai camera phải cùng thấy **ít nhất 4 điểm băng keo**. Các điểm không thẳng hàng và rải dọc hành lang, để tính được homography mặt sàn.
- Tránh những thứ sau trong nền:
  - Kính hoặc gương, vì có phản chiếu.
  - Nắng chiếu thẳng hoặc đèn nhấp nháy.
  - Cửa đóng mở liên tục.
- Chụp lại ảnh vị trí chân máy để quay lại đúng như cũ khi cần.

## 4. Quy trình mỗi lượt quay

1. Bấm quay cả hai máy.
2. Quay **3–5 giây cảnh trống**, không có ai trong khung. Đoạn này dùng để khởi tạo ảnh nền.
3. Một người giơ **thẻ xám** ở giữa khu vực trong khoảng 2 giây.
4. **Vỗ tay một cái thật rõ** ở chỗ cả hai camera cùng thấy. Tiếng và hình vỗ tay dùng để đồng bộ hai video.
5. Diễn kịch bản trong 20–40 giây.
6. Quay thêm 2 giây cảnh trống, rồi dừng cả hai máy.
7. Ghi vào sổ: tên file trên mỗi máy, kịch bản nào, lượt dev hay test.

## 5. Danh sách kịch bản

Tất cả đều quay trong hành lang, theo bố trí ở mục 3.

| Kịch bản | Nội dung | Kiểm tra phần nào | Số lượt |
|---|---|---|---|
| S1 | 4 người đi thưa theo hai chiều, mỗi người một làn | C, D, G cơ bản | 2 (dev, test) |
| S2 | Đi ngược chiều, đổi làn, cắt nhau và che nhau | D (blob gộp và tách) | 2 |
| S3 | Một người **đứng yên ít nhất 5 giây**; người khác đi ngang trước mặt | C, D (trạng thái `static`) | 2 |
| S4 | Trong 4 người có **2 người mặc áo cùng tông màu**; hai người này đi ngược chiều cắt nhau | D, G (không dựa vào màu) | 2 |
| S5 | 2 người **đi sát nhau thành nhóm** một đoạn rồi tách ra | D (nhóm đi cùng) | 2 |

Tối thiểu cần 5 kịch bản × 2 lượt × 2 camera = **20 video**.

Lưu ý cho báo cáo: dev và test dùng **cùng 4 người**, nên kết quả Re-ID sau khi chỉnh trọng số trên dev có thể lạc quan (XLA 3.1).

## 6. Sau khi quay

1. **Chuyển sang frame rate cố định.** Điện thoại hay quay frame rate thay đổi (VFR), làm hai video lệch dần theo thời gian (XLA 3.6). Giữ lại âm thanh để còn tìm tiếng vỗ tay:
   ```bash
   ffmpeg -i raw.mp4 -vf fps=30 -c:v libx264 -crf 18 -c:a aac video.mp4
   ```
2. **Xếp file vào thư mục:**
   ```
   data/canh/<ten_canh>/<cam_a|cam_b>/video.<đuôi>      (ví dụ data/canh/hanh_lang_s2_dev/)
   ```
   Cách nhanh: chép video gốc vào `data/video_moi/` rồi chạy `python -m tools.tao_canh` (xem [data/README.md](../data/README.md)).
   Không đưa video lên GitHub.
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
