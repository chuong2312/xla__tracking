# Câu hỏi cho thầy và câu trả lời

> **Giai đoạn 0 đã xong:** thầy đã trả lời cả 8 câu, ngày 2026-11-21.
> Câu trả lời ảnh hưởng tới thiết kế thế nào thì xem [quyet_dinh_thiet_ke.md](quyet_dinh_thiet_ke.md) và [de_xuat_phuong_an.md](de_xuat_phuong_an.md).

## Tóm tắt

| # | Câu hỏi | Thầy trả lời | Nhóm làm gì |
|---|---|---|---|
| 1 | Dùng AI viết code, có cần khai báo không | Thoải mái | Dùng AI, nhưng vẫn phải tự giải thích được mọi dòng code (QĐ-10) |
| 2 | MOG2 | Được dùng | Cài cả MOG2 và trừ nền trung vị, chọn cái mặc định theo kết quả trên dev (QĐ-03) |
| 3 | Auto SIFT | Được dùng | Thầy không nói rõ là nghĩa nào, nên làm cả `sift_auto` lẫn ASIFT và so với SIFT thường (QĐ-14) |
| 4 | Dùng thư viện có sẵn | Được dùng | Dùng hàm của OpenCV/SciPy; tự đặt tham số và giải thích được bên trong (QĐ-11) |
| 5 | Kalman | Được dùng | Dùng `cv2.KalmanFilter` trong tracker mới (QĐ-05) |
| 6 | Đếm người IN/OUT | **Bắt buộc** | Ngày 2026-11-22 nhóm quyết định **bỏ** phần này (QĐ-21). Như vậy là trái yêu cầu của thầy, nên cần trao đổi lại với thầy. Code cũ còn trong commit `827121c` |
| 7 | Số camera, số người | Tối đa 4 camera cố định | Nhóm chọn 2 camera, 4 người (QĐ-13). Thầy không nói gì về số người. |
| 8 | Homography mặt sàn | Được dùng | Chỉ dùng làm thông tin hình học **bổ trợ**; kết quả chỉ dùng ngoại hình được báo cáo riêng (QĐ-02) |

## Ghi chú cho từng câu

**Câu 2 (MOG2).** Thầy cho dùng không có nghĩa là được phép nói "MOG2 không phải ML".
- MOG2 là mô hình nền hỗn hợp Gaussian thích nghi.
- Nó chỉ phát hiện foreground (vùng khác nền), chứ không trực tiếp xác định pixel nào là người.

Khi bị hỏi, trả lời theo [hoi_dap_bao_ve.md](hoi_dap_bao_ve.md), không khẳng định đơn giản là có hay không.

**Câu 3 (Auto SIFT).** SIFT và ASIFT là hai thuật toán khác nhau.
- Trong OpenCV, ASIFT là hàm `cv2.AffineFeature_create`.
- Không phải phiên bản OpenCV nào cũng có hàm này. Nhóm đã chạy thử thành công trên OpenCV 5.1.0, nên phải ghim đúng phiên bản này trong `requirements.txt`.

**Câu 7 (số người).** Thầy chỉ giới hạn số camera. Nhóm chọn 4 người để có nhiều cặp ghép hơn khi đánh giá Re-ID.

**Câu 8 (homography).** Homography dựa trên giả định mặt sàn gần phẳng, nên có thể sai, ví dụ khi:
- chân người bị che,
- box lệch,
- hai video đồng bộ lệch,
- hai người đứng sát nhau.

Không được mặc định là homography luôn ghép đúng người.

## Câu hỏi gốc (để tra lại)

1. Em có được dùng trợ lý AI để hỗ trợ viết code không? Nếu được thì cần ghi nhận trong báo cáo như thế nào?
2. MOG2 có bị tính là học máy không? Nếu có thì trừ nền bằng trung vị kết hợp trung bình trượt có được không?
3. Yêu cầu "auto SIFT" là ASIFT hay là SIFT tự chỉnh ngưỡng theo số keypoint?
4. Em có bắt buộc tự cài các thuật toán (Lucas-Kanade, Hungarian, RANSAC, trừ nền) không, hay được gọi hàm của OpenCV/SciPy?
5. Em có được dùng bộ lọc Kalman không?
6. Phần đếm người qua cửa có bắt buộc không?
7. Nhóm dự kiến quay 3 người với 2 camera chồng vùng nhìn; như vậy có đủ không?
8. Em dùng thêm homography mặt sàn để hỗ trợ ghép người giữa hai camera, và báo cáo riêng kết quả chỉ dùng ngoại hình. Như vậy có đúng yêu cầu không?

Không cần hỏi về HOG+SVM: đó là ML, và phương án mới không dùng HOG (XLA 1).
