# Nhật ký quyết định thiết kế

> Mỗi quyết định ghi lại: bối cảnh, quyết định, lý do, hệ quả và tham chiếu XLA.md.
> Khi bảo vệ, câu "vì sao em chọn cách này?" được trả lời từ file này.
> Có ba trạng thái:
> - **đã chốt:** nhóm đã đồng ý.
> - **đề xuất:** chờ nhóm duyệt.
> - **chờ thầy:** phụ thuộc vào [cau_hoi_thay.md](cau_hoi_thay.md). Từ 2026-11-21 không còn quyết định nào ở trạng thái này, vì thầy đã trả lời hết.

## QĐ-01: Quay lại dữ liệu bằng chân máy
- **Trạng thái:** đã chốt (2026-11-21).
- **Bối cảnh:** video `data/real/dev/S1/cam_a` được quay cầm tay. Đo bằng `cv2.phaseCorrelate` thì khung hình trôi khoảng 25 px (ở độ rộng 640) sau 15 giây, kèm theo xoay. Trừ nền vì thế bắt toàn bộ cạnh tường, sinh tới 13 detection mỗi frame và 134 track ID.
- **Quyết định:** quay lại toàn bộ dữ liệu theo [ke_hoach_quay_video.md](ke_hoach_quay_video.md).
- **Lý do:** mọi phương pháp trừ nền đều giả định camera cố định. Chỉnh tham số không sửa được lỗi của dữ liệu. Ổn định ảnh bằng phần mềm thì tốn thêm khoảng 1 tuần và vẫn có rủi ro.
- **Hệ quả:** video S1 cũ và các kết quả trong `outputs/real_dev_S1/` chỉ còn giá trị minh họa cho lý do phải quay lại.

## QĐ-02: Hai camera chồng vùng nhìn
- **Trạng thái:** đã chốt (2026-11-21). Thầy cho dùng homography (câu 8 trong [cau_hoi_thay.md](cau_hoi_thay.md)). Số camera chốt là 2, xem QĐ-13.
- **Bối cảnh:** XLA 2.11 yêu cầu chốt cách bố trí camera, vì bố trí quyết định toàn bộ chiến lược Re-ID.
- **Quyết định:** hai camera cùng nhìn một khu vực từ hai góc lệch nhau 45–90°.
- **Lý do:**
  - Có thể dùng homography mặt sàn làm **thông tin hình học bổ trợ**. Cách này đúng chủ đề xử lý ảnh, và RANSAC được dùng đúng chỗ vì mặt sàn gần như là một mặt phẳng.
  - Không được mặc định homography luôn đúng. Nó dựa trên giả định mặt sàn gần phẳng, và sẽ sai khi chân bị che, box lệch, hai video đồng bộ lệch, hoặc hai người đứng sát nhau.
  - Ground truth Re-ID dễ gán hơn.
- **Hệ quả:**
  - Phải đồng bộ thời gian chính xác giữa hai video (vỗ tay, chuyển sang frame rate cố định).
  - Phải dán điểm mốc trên sàn.
  - Phải **báo cáo tách riêng** kết quả "chỉ ngoại hình" và "ngoại hình + hình học".

## QĐ-03: Cài cả trừ nền trung vị lẫn MOG2, chọn cái mặc định theo kết quả trên dev
- **Trạng thái:** đề xuất. Cập nhật ngày 2026-11-21, sau khi thầy cho dùng MOG2 (câu 2).
- **Bối cảnh:**
  - MOG2 là mô hình nền hỗn hợp Gaussian thích nghi, dùng để phát hiện foreground. Nó không trực tiếp xác định pixel nào là người. XLA 1 xếp MOG2 vào vùng xám, nhưng thầy đã cho dùng.
  - MOG2 của OpenCV dùng chung một `learningRate` cho cả ảnh, nên không cập nhật nền có chọn lọc được. Hệ quả là người đứng yên có thể dần bị "nuốt" vào nền (XLA 2.2).
- **Quyết định:** interface `BackgroundModel` có hai cài đặt:
  - `median_running`: khởi tạo nền bằng trung vị, cập nhật có chọn lọc bằng `accumulateWeighted` kèm mask, khử bóng bằng luật HSV (XLA 1.1).
  - `mog2`: tái dùng code có sẵn trong `src/detector.py`.

  Cài đặt nào làm mặc định sẽ chọn theo kết quả ablation trên dev, dựa trên hai chỉ số: tỉ lệ frame đếm đúng số người, và kết quả riêng trên S3.
- **Lý do:**
  - Chưa có dữ liệu thật thì chưa biết cách nào tốt hơn.
  - `median_running` có lợi thế khi người đứng yên. MOG2 có sẵn phát hiện bóng và chịu được nền dao động.
  - Bản thân phép so sánh cũng là một kết quả để đưa vào báo cáo.
- **Hệ quả:** vẫn phải quay 3–5 giây cảnh trống ở đầu mỗi video, vì `median_running` cần đoạn này để khởi tạo nền.

## QĐ-04: Xóa HOG+SVM
- **Trạng thái:** đề xuất (bắt buộc theo XLA 1).
- **Bối cảnh:** `src/detector.py` có tùy chọn `use_hog_baseline`, dùng bộ SVM đã được huấn luyện sẵn của OpenCV.
- **Quyết định:** xóa hẳn khỏi code và khỏi config, không chỉ tắt đi.
- **Lý do:** dù đang tắt, hội đồng đọc code vẫn thấy SVM, tức là ML.

## QĐ-05: Viết lại tracker theo Kalman + LK
- **Trạng thái:** đã chốt (2026-11-21). Thầy cho dùng Kalman (câu 5).
- **Bối cảnh:** tracker LK cũ dài 777 dòng, khoảng 1/3 là diagnostics, có logic ownership nhiều nhánh. Nó vẫn đổi ID 3 lần trong test giao cắt dù dùng detection chuẩn. Nhóm cần giải thích được từng bước khi bảo vệ.
- **Quyết định:**
  - Kalman vận tốc hằng dùng `cv2.KalmanFilter`; nhóm tự đặt các ma trận `F`, `H`, `Q`, `R`.
  - LK đóng vai trò một phép đo chuyển động, lấy điểm trên vùng box ∩ erode(foreground).
  - Máy trạng thái rõ ràng: `tentative / confirmed / merged / static / lost`.
  - Phân loại blob gộp và blob tách trước khi chạy Hungarian.
  - Khi hai người tách ra, ghép lại bằng nhiều tín hiệu.
- **Lý do:** Kalman cho dự đoán khi bị che (tính liên tục chuyển động, XLA 2.4). Máy trạng thái thay cho các cờ rời rạc. Mỗi file làm một việc nên dễ giải thích.
- **Hệ quả:**
  - Tracker cũ được giữ trong git bằng một tag.
  - `tests/test_tracker_crossing.py` được tái sử dụng làm test chấp nhận.
  - Định dạng `tracks.csv` giữ nguyên.

## QĐ-06: Ràng buộc dọc là mặc định, RANSAC chỉ để ablation
- **Trạng thái:** đề xuất.
- **Bối cảnh:** `estimateAffinePartial2D` giả định quan hệ đồng dạng, không đúng với người là vật không cứng. Khi chỉ có ít cặp khớp, RANSAC luôn "tìm được" một mô hình (XLA 2.8).
- **Quyết định:**
  - Yêu cầu tối thiểu 6–8 cặp khớp.
  - Ràng buộc mặc định là `|y_A/H − y_B/H| < δ`.
  - RANSAC chỉ là một nhánh trong ablation.
- **Hệ quả:** nếu RANSAC làm giảm rank-1 thì đó cũng là một phát hiện đáng báo cáo.

## QĐ-07: Resize giữ tỉ lệ về 640×360
- **Trạng thái:** đề xuất.
- **Bối cảnh:** config cũ dùng 640×480 (4:3), làm méo video 16:9 và sai lệch mọi ngưỡng tính bằng pixel (XLA 2.15).
- **Quyết định:** resize giữ tỉ lệ. Ngưỡng pixel ghi ở độ phân giải tham chiếu và quy đổi bằng `scaled()`.

## QĐ-08: Hiệu chỉnh tham số bằng quét lưới, không dùng chữ "train"
- **Trạng thái:** bắt buộc (XLA 1, 5).
- **Quyết định:**
  - Trọng số và ngưỡng chỉ chọn bằng quét lưới thủ công trên dev, có ghi lại.
  - Không dùng hồi quy hay tối ưu gradient để "fit".
  - Trong báo cáo dùng các từ "hiệu chỉnh tham số", "tập hiệu chỉnh", "tập đánh giá".

## QĐ-09: Làm bản tối thiểu chạy từ đầu đến cuối trước, cổng nghiệm thu có hạn thời gian
- **Trạng thái:** đề xuất.
- **Bối cảnh:**
  - Công sức trước đây dồn vào tối ưu tracker trên video tổng hợp qua 5 vòng sửa và 2 vòng audit, trong khi Re-ID (phần chính của đề tài) chưa có.
  - Cổng nghiệm thu cứng có thể chặn cả dự án (XLA 4.2).
- **Quyết định:**
  - Thứ tự ưu tiên: C (bản tối thiểu) → D → F (chỉ SIFT) → G → H (XLA 4.4).
  - Mỗi cổng có giới hạn tối đa 5 ngày. Quá hạn thì ghi nhận là hạn chế rồi làm tiếp.

## QĐ-10: Ưu tiên code dễ hiểu và có tài liệu cho từng module
- **Trạng thái:** đã chốt (2026-11-21).
- **Bối cảnh:** nhóm viết code cùng AI (thầy không cấm, xác nhận ngày 2026-11-21). Vì vậy mục tiêu số 1 là hiểu và bảo vệ được.
- **Quyết định:**
  - Áp dụng quy tắc code và quy trình bàn giao ở mục 6 của [de_xuat_phuong_an.md](de_xuat_phuong_an.md).
  - Mỗi module bàn giao kèm tài liệu, lệnh `--explain` xuất ảnh minh họa, câu hỏi bảo vệ và một buổi tự kiểm tra.

## QĐ-11: Dùng hàm có sẵn của OpenCV và SciPy
- **Trạng thái:** đã chốt. Thầy cho phép (2026-11-21).
- **Quyết định:** dùng các hàm có sẵn thay vì tự cài: `calcOpticalFlowPyrLK`, `KalmanFilter`, `linear_sum_assignment`, `findHomography`, `createBackgroundSubtractorMOG2`, SIFT/ORB… Nhóm chỉ tự viết những phần ngắn và là ý tưởng riêng của nhóm: trừ nền trung vị, gating, máy trạng thái, cách tính điểm tương đồng.
- **Lý do:** dành thời gian cho phần chính là Re-ID và đếm qua cửa.
- **Hệ quả:** vẫn phải giải thích được thuật toán bên trong từng hàm. Hàm thư viện nào được dùng thì có mục giải thích trong tài liệu của module tương ứng.

## QĐ-12: Đếm người IN/OUT là bắt buộc
- **Trạng thái:** đã chốt. Thầy yêu cầu (2026-11-21).
- **Quyết định:**
  - Phần E (`door.py`) nằm trong phạm vi bắt buộc.
  - Cảnh cửa là kịch bản bắt buộc trong [ke_hoach_quay_video.md](ke_hoach_quay_video.md), và phải có lúc 2 người qua cửa cùng lúc theo hai chiều ngược nhau.

## QĐ-13: 2 camera, 4 người
- **Trạng thái:** đã chốt (nhóm, 2026-11-21). Thầy cho tối đa 4 camera cố định và không nói gì về số người.
- **Quyết định:** dùng 2 camera chồng vùng nhìn, mỗi cảnh có 4 người. Config và code Re-ID vẫn thiết kế cho tối đa 4 camera, ghép theo từng cặp camera.
- **Lý do:**
  - 2 camera là cách bố trí đơn giản nhất để đồng bộ, tính homography và gán nhãn.
  - 4 người cho nhiều cặp ghép hơn 3 người, nên kết quả ít bị "nhảy bậc" hơn (XLA 3.2). Cụ thể, khi hai camera đều phát hiện đủ người và ghép một-một trọn vẹn: với 3 người, số cặp đúng chỉ có thể là 0, 1 hoặc 3; với 4 người là 0, 1, 2 hoặc 4.
- **Hệ quả:** che khuất nhiều hơn, và công gán nhãn tăng khoảng 1/3 (còn khoảng 10,5–17,5 giờ).

## QĐ-14: Auto SIFT: làm cả hai nghĩa rồi so sánh
- **Trạng thái:** đã chốt (nhóm, 2026-11-21). Thầy cho dùng auto SIFT nhưng không nói rõ là nghĩa nào.
- **Quyết định:** bảng ablation của phần F/G có ba nhánh, nhánh nào cũng chỉ giữ N keypoint mạnh nhất:
  - SIFT thường: `SIFT(nfeatures=N)`.
  - `sift_auto`: khi crop có quá ít keypoint thì hạ `contrastThreshold` rồi chạy lại (XLA 2.18).
  - ASIFT: `cv2.AffineFeature_create(cv2.SIFT_create())` (Morel và Yu, 2009).
- **Lý do:** hai nghĩa giải hai vấn đề khác nhau. `sift_auto` giải chuyện crop có quá ít keypoint. ASIFT giải chuyện hai camera nhìn lệch góc nhau.
- **Hệ quả:**
  - Không phải phiên bản OpenCV nào cũng có ASIFT. Nhóm đã chạy thử thành công trên OpenCV 5.1.0, nên phải ghim đúng phiên bản này.
  - ASIFT chậm hơn và sinh nhiều keypoint hơn hẳn. Trên crop 128×256, ASIFT ra 8.206 keypoint trong 64 ms, còn SIFT ra 501 keypoint trong 19 ms. Vì vậy ASIFT chỉ chạy trên các crop của gallery.
