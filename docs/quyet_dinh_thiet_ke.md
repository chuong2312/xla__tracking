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
  - `mog2`: tái dùng code có sẵn trong `legacy/detector.py`.

  Cài đặt nào làm mặc định sẽ chọn theo kết quả ablation trên dev, dựa trên hai chỉ số: tỉ lệ frame đếm đúng số người, và kết quả riêng trên S3.
- **Lý do:**
  - Chưa có dữ liệu thật thì chưa biết cách nào tốt hơn.
  - `median_running` có lợi thế khi người đứng yên. MOG2 có sẵn phát hiện bóng và chịu được nền dao động.
  - Bản thân phép so sánh cũng là một kết quả để đưa vào báo cáo.
- **Hệ quả:** vẫn phải quay 3–5 giây cảnh trống ở đầu mỗi video, vì `median_running` cần đoạn này để khởi tạo nền.

## QĐ-04: Xóa HOG+SVM
- **Trạng thái:** đề xuất (bắt buộc theo XLA 1).
- **Bối cảnh:** `legacy/detector.py` có tùy chọn `use_hog_baseline`, dùng bộ SVM đã được huấn luyện sẵn của OpenCV.
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
  - `tests/legacy/test_tracker_crossing.py` được tái sử dụng làm test chấp nhận.
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

## QĐ-12: Đếm người IN/OUT là bắt buộc *(ĐÃ BỎ theo QĐ-21)*
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

## QĐ-15: File đáp án có thêm cột `occluded` và `keyframe`
- **Trạng thái:** đã làm (2026-11-21).
- **Bối cảnh:** XLA 3.3 yêu cầu:
  - ghi rõ người bị che hay không;
  - chỉ chấm precision/recall của detector trên frame gán tay;
  - bỏ qua frame bị che nặng.
- **Quyết định:** đáp án có các cột `frame,gt_id,x,y,w,h,occluded,keyframe`.
  - `occluded`: 0 là không che, 1 là che một phần, 2 là che nặng.
  - `keyframe`: 1 là box gán tay, 0 là box nội suy. Frame nội suy lấy mức che lớn hơn của hai keyframe hai bên.
  - Công cụ gán nhãn có thêm phím `o` để đổi mức che.
- **Hệ quả:**
  - `detection_pr` chỉ chấm frame mà mọi box đều gán tay và không box nào bị che nặng.
  - File đáp án cũ (không có hai cột này) vẫn đọc và chấm được như trước.

## QĐ-16: Video tổng hợp v2 vẽ theo tọa độ sàn và homography *(ĐÃ BỎ theo QĐ-31: chỉ dùng video thật)*
- **Trạng thái:** đã làm (2026-11-21).
- **Bối cảnh:** chưa có video thật. Video tổng hợp cũ chỉ có đáp án cho một camera, các hộp màu trên nền phẳng, và chỉ 3 "người".
- **Quyết định:**
  - Đặt 4 người trên một mặt sàn tính bằng mét; mỗi camera là một homography sàn → ảnh.
  - Vẽ người ở xa trước để có che khuất tự nhiên.
  - Dùng bản đồ "ai ở pixel nào" để tính box phần nhìn thấy và mức che.
  - 5 kịch bản: `s1`, `s2`, `s3`, `s4`, `door`.
- **Lý do:** có đáp án chính xác cho cả hai camera, cho Re-ID (ai là ai) và cho sự kiện qua cửa, cùng với homography A → B biết trước để kiểm tra công cụ click điểm sàn.
- **Hệ quả:** người là hình chữ nhật đi thẳng đều, nên kết quả trên video tổng hợp chỉ dùng để kiểm tra code. Tham số phải hiệu chỉnh lại trên video thật.

## QĐ-17: Cảnh cửa dùng bố trí "hành lang + cửa" *(ĐÃ BỎ theo QĐ-21)*
- **Trạng thái:** đã chốt cho cảnh cửa (2026-11-21). Các kịch bản S1–S4 vẫn theo QĐ-02 (hai camera cùng nhìn một khu vực). Nếu sau này muốn dùng bố trí này cho cả đồ án thì chỉ cần sửa kế hoạch quay; code đã hỗ trợ cả hai.
- **Bối cảnh:** nhóm đề xuất bố trí giống camera an ninh thật: camera A ở đầu hành lang thấy người đi tới rồi rẽ vào cửa; camera B gắn đối diện cửa, thấy người đi qua cửa.
- **Quyết định:** quay cảnh cửa theo bố trí này ([ke_hoach_quay_video.md](ke_hoach_quay_video.md), mục 3b). Video tổng hợp có kịch bản `hanh_lang_cua` mô phỏng đúng bố trí này.
- **Lý do:**
  - Sát thực tế.
  - Camera B nhìn thẳng cửa nên đếm VÀO/RA tốt.
  - Có thêm một cách nhận ra cùng một người mà không cần nhìn quần áo (QĐ-18).
- **Hệ quả:**
  - Hai camera chỉ chung vùng quanh cửa, nên homography chỉ dùng được ở vùng đó.
  - Camera A nhìn cửa rất nghiêng, nên đếm ở camera A kém tin cậy hơn camera B.
  - Đồng bộ giờ trở thành bắt buộc.

## QĐ-18: Ghép người giữa hai camera bằng thời gian qua cửa *(ĐÃ BỎ theo QĐ-21)*
- **Trạng thái:** đã làm (2026-11-21), `src/door_matching.py`.
- **Bối cảnh:** XLA 2.11 gợi ý dùng thời gian làm ràng buộc khi các camera không cùng nhìn một khu vực. Với bố trí QĐ-17, thời điểm qua cửa là tín hiệu rất rõ.
- **Quyết định:** hai lượt qua cửa **cùng chiều**, lệch nhau không quá `reid.door_event_max_gap_s` giây, được phép ghép. Ghép một-một bằng Hungarian để tổng độ lệch nhỏ nhất.
- **Lý do:** đơn giản, không dùng màu, nên giải được cả trường hợp hai người mặc giống nhau.
- **Hệ quả:**
  - Bó tay khi hai người qua cửa gần như cùng lúc; khi đó kết hợp thêm ngoại hình và homography.
  - Trong báo cáo, kết quả "chỉ dùng thời gian" được báo riêng, giống cách báo riêng kết quả homography.

## QĐ-19: Cách tính và gộp điểm giống nhau khi nhận dạng lại
- **Trạng thái:** đã làm (2026-11-21). Các con số phải hiệu chỉnh lại trên dev thật.
- **Quyết định:**
  - **Điểm keypoint** = 0 nếu có ít hơn 6 cặp khớp tốt, ngược lại `min(1, n/30)`. Lấy cặp ảnh mẫu tốt nhất giữa hai gallery.
  - **Điểm màu** = 1 − trung bình khoảng cách Bhattacharyya của thân trên và thân dưới.
  - **Điểm sàn** = trung bình `exp(−d²/2σ²)`.
  - **Gộp** bằng trung bình có trọng số 0,4 / 0,4 / 0,2, chỉ tính các điểm đang có.
  - Ghép bằng Hungarian; **ngưỡng 0,3**.
  - Luôn báo riêng "chỉ ngoại hình" và "ngoại hình + sàn".
- **Lý do:**
  - Đổi số cặp khớp thành điểm 0–1 có "mức đủ", để không bị phụ thuộc vào tổng số keypoint.
  - Ngưỡng 0,3 nằm giữa điểm của cùng người (0,42–0,94) và khác người (≤ 0,14) trên video tổng hợp.

## QĐ-20: Giai đoạn 2 bắt đầu; tracker mới đặt trong file riêng, tracker cũ giữ lại
- **Trạng thái:** đã làm (2026-11-21). Nhóm yêu cầu làm phần hệ thống trước, chưa cần dữ liệu thật.
- **Quyết định:**
  - Hệ thống mới gồm `src/detect/detection.py`, `src/tracking/people_tracker.py`, `src/pipeline.py` và `run.py`.
  - Tracker cũ (`legacy/tracker.py`) cùng detector cũ được giữ nguyên, để báo cáo có thể so sánh cũ với mới.
  - Hệ thống được thử trên video tổng hợp v2.
- **Ba quy tắc của tracker**, rút ra khi thử ([11_tracker_moi.md](11_tracker_moi.md)):
  1. Box mới to hoặc nhỏ bất thường **và** chồng lên người khác thì không lấy vị trí, kích thước từ box đó.
  2. Chỉ gộp các mảnh nằm trên dưới nhau.
  3. Khi hai người dính nhau, chỉ người đứng trước giữ Lucas-Kanade. Thí nghiệm so sánh: MOTA 0,758, IDF1 0,84, tốt nhất trong 3 cách thử.
- **Hệ quả:** `src/evaluation/metrics.py` giờ chấm cả các trạng thái `merged` và `static` (`SCORED_TRACK_STATES`), vì tracker mới vẫn báo vị trí người trong hai trạng thái đó.

## QĐ-21: Bỏ phần đếm VÀO/RA qua cửa
- **Trạng thái:** đã làm (nhóm quyết định, 2026-11-22).
- **Bối cảnh:** nhóm muốn tập trung vào video hành lang có nhiều người đi lại.
- **Quyết định:** bỏ hẳn:
  - đếm VÀO/RA (`src/door.py`);
  - ghép hai camera theo thời gian qua cửa (`src/door_matching.py`);
  - các cảnh `door`, `hanh_lang_cua`;
  - tài liệu 05, 07.
  Các mục QĐ-12, QĐ-17, QĐ-18 hết hiệu lực.
- **Rủi ro:** ngày 2026-11-21 thầy đã trả lời rằng đếm VÀO/RA là **bắt buộc** (câu 6 trong [cau_hoi_thay.md](cau_hoi_thay.md)). Bỏ phần này là trái yêu cầu đó. **Nhóm cần trao đổi lại với thầy.**
- **Lấy lại khi cần:** toàn bộ code, test và tài liệu của phần cửa còn trong commit `827121c` trên nhánh `redesign`.

## QĐ-22: Chỉ một bố trí: hành lang, hai camera ở cùng một đầu *(phần video máy vẽ ĐÃ BỎ theo QĐ-31; bố trí vẫn dùng cho kế hoạch quay)*
- **Trạng thái:** đã làm (nhóm chốt, 2026-11-22).
- **Quyết định:**
  - Hành lang rộng 3 m, dài 10 m.
  - Hai camera ở cùng một đầu (góc trái, góc phải), cùng nhìn dọc vào trong.
  - Mỗi cảnh có 4 người.
  - Video tổng hợp có 5 kịch bản: `s1` đi thưa, `s2` cắt nhau, `s3` đứng yên, `s4` mặc giống, `s5` đi thành nhóm.
  - Bố trí `san_chung` cũ cũng bỏ.
- **Lý do:** hai camera cùng thấy mặt trước người, nên vẫn dùng được vị trí sàn (QĐ-02). Bố trí này cũng giống camera an ninh hành lang thật.
- **Hệ quả:** trong hành lang nhìn dọc, nhiều người đi ngược chiều sẽ chồng lên nhau lâu, nên khó giữ đúng tên ([11](11_tracker_moi.md), [12](12_he_thong.md)).

## QĐ-23: Nối các đoạn track bị đứt sau khi bám người
- **Trạng thái:** đã làm (2026-11-22), `src/tracking/tracklet_linking.py`.
- **Quyết định:**
  - Nối đoạn sau với đoạn trước khi đoạn sau bắt đầu sau đoạn trước không quá 120 frame, ở gần chỗ đoán theo vận tốc (không quá 3 lần đường chéo box), và có màu áo giống.
  - Ghép bằng Hungarian.
  - Báo MOTA/IDF1 cả trước lẫn sau khi nối.
- **Kết quả:** IDF1 trung bình tăng 0,783 → 0,796 ([13](13_noi_tracklet.md)). Giúp ít, vì lỗi chính là tráo tên chứ không phải track bị đứt.

## QĐ-24: Sửa lỗi tráo tên khi nhiều người dính nhau lâu
- **Trạng thái:** đã làm (2026-11-22).
- **Nguyên nhân tìm được:** người ở mép cụm gộp chập chờn quanh ngưỡng 60% "trong cụm / ngoài cụm", bị tính là mất dấu rồi bị xóa. Khi người đó hiện ra, track của người khác (vẫn còn khuất) "cướp" box của họ.
- **Quyết định:**
  - **Ngưỡng có độ trễ:** vào cụm cần 60%, ở lại cụm chỉ cần 30% (`merge_keep_coverage`).
  - **Chặn ghép lại khác màu:** độ giống màu < 0,5 thì cấm (`recover_min_color_similarity`).
- **Đã thử nhưng không giữ** (bảng ở [11](11_tracker_moi.md)):
  - thêm màu vào bước ghép bình thường;
  - cập nhật màu và ảnh mẫu từ từ;
  - đổi trọng số khi ghép lại;
  - so khớp ảnh mẫu NCC trong cụm.

  Các công tắc vẫn còn trong config để chạy lại thí nghiệm.
- **Kết quả:** IDF1 trung bình 0,783 → 0,806. Trên toàn hệ thống: `s2` cam B 0,66 → 0,78, `s4` cam B 0,69 → 0,78, nhận dạng lại ở `s4` từ 1/4 lên 2/4 (chỉ ngoại hình) và 4/4 (có vị trí sàn).

## QĐ-25: Dùng MOG2 làm mô hình nền mặc định
- **Trạng thái:** đã làm (2026-11-22), `detection.background: mog2`.
- **Bối cảnh:** lần đầu chạy trên video thật (EPFL "4p", camera c0/c1, 360×288, 25 fps). Không có cảnh trống ở đầu video, ánh sáng nhấp nháy.
- **Đo được** (tỉ lệ pixel bị coi là foreground, c1, 1800 frame đầu; càng thấp càng ít báo nhầm):
  - trừ nền trung vị: 15,5%; báo nhầm cả mảng tường và thảm;
  - trừ nền trung vị kèm bù độ sáng (`exposure_compensation`): 14,8%, gần như không đỡ;
  - MOG2: 5,0%.
- **Quyết định:** chọn MOG2 làm mặc định. Trừ nền trung vị vẫn giữ để so sánh; `exposure_compensation` mặc định tắt.
- **Đánh đổi:** trên video tổng hợp, MOG2 cho MOTA cao hơn nhưng nhận dạng lại ở `s2` kém hơn ([12](12_he_thong.md)). MOG2 cập nhật nền cả ảnh, nên người đứng yên lâu có thể bị "nuốt" dần vào nền.

## QĐ-26: Gộp mảnh người dựa trên mô hình chiều cao theo vị trí chân
- **Trạng thái:** đã làm (2026-11-22), `src/detect/height_model.py`, `src/detect/detection.py`.
- **Vấn đề:** trên EPFL, người mặc áo đen đứng trước ghế hay tường tối bị mất cả phần thân trong mask. Bước tìm người trả về hai mảnh, đầu và chân, và tracker tạo hai track cho cùng một người. Khoảng một nửa số khung là mảnh người (chiều cao trung vị chỉ bằng nửa chiều cao người nguyên vẹn).
- **Đã thử nhưng bỏ** (trung bình 5 cảnh tổng hợp, sau khi nối tracklet):
  - gộp mọi mảnh xếp trên – dưới: IDF1 0,82 → 0,77; nhận dạng lại `s2` còn 1/4;
  - bỏ track mảnh trong tracker (`fragment_max_vertical_overlap`, để lại trong config, mặc định tắt): IDF1 0,81 → 0,78.

  Lý do chung: trong hành lang nhìn dọc, người ở xa nằm ngay trên đầu người ở gần, nên "trên – dưới" không phân biệt được "mảnh của một người" với "hai người đứng trước – sau".
- **Quyết định:**
  - Ước lượng chiều cao người mong đợi `h(y_chân) = a·y_chân + b` từ 1500 frame đầu của chính video. Lấy phân vị 90% của chiều cao box trong từng khoảng y_chân, rồi kẻ đường thẳng qua các điểm đó bằng **Theil–Sen** (độ dốc = trung vị độ dốc của mọi cặp điểm). Lần đầu dùng bình phương tối thiểu thì khoảng sát mép dưới ảnh (người bị cắt chân) kéo cả đường xuống thấp; trung vị không bị kéo như vậy.
  - Chỉ gộp hai mảnh thẳng hàng khi box gộp cao không quá `1,15 × h(y_chân)`.
- **Không phải học máy:** chỉ là thống kê (phân vị, trung vị) và kẻ đường thẳng qua khoảng 10 điểm; không có nhãn, không có mô hình phân loại.
- **Kết quả:**
  - Tổng hợp (MOG2, trung bình 5 cảnh): MOTA 0,718 → 0,723, IDF1 0,814 → 0,822.
  - EPFL (không có đáp án, chỉ đếm): số track đang chạy mỗi frame giảm 5,9 → 4,8 (c0) và 5,9 → 4,6 (c1), trong cảnh có tối đa 4 người. Vẫn còn thừa track và đổi ID nhiều.
- **Ca còn sai:** hai người cùng mặc áo đen đứng chồng trước – sau. Box gộp "đầu người sau + chân người trước" chỉ cao hơn một người vài pixel nên vẫn bị gộp (`outputs/explain/chieu_cao/2_gop_manh.png`).

## QĐ-27: Chỉ tạo track mới từ khung đủ cao một người
- **Trạng thái:** đã làm (2026-11-22), `src/detect/height_model.py` (`tall_enough`), `src/tracking/people_tracker.py` (`_create_tracks`).
- **Bối cảnh:** sau QĐ-26, EPFL vẫn có khoảng 4,7 track mỗi frame (cảnh tối đa 4 người) và gần 200 ID mỗi camera. Phân tích các track:
  - khoảng 250 track mỗi camera chết khi còn `confirmed`, tuổi trung vị chỉ 36 frame (1,4 giây);
  - bỏ các track rời khỏi mép ảnh, còn khoảng 200 track; 11 frame trước khi chết, những track này đã không còn điểm LK, và ở frame cuối 85% box gần như trống.

  Đây là **track trùng** sinh ra từ một mảnh người (đầu hoặc chân). Khi người đó lại được tìm thấy nguyên vẹn, track chính nhận khung, còn track mảnh trôi vào chỗ trống rồi chết.
- **Quyết định:** detection chỉ được tạo track mới khi `h ≥ 0,6 · h_mong_đợi(y_chân)` (`new_track_min_height_ratio`), trừ khi box chạm mép ảnh (`new_track_border_px` = 5), vì người đang bước vào khung hình thì bị cắt. Khung mảnh vẫn được ghép với track đã có như cũ.
- **Hiệu chỉnh tham số** (tổng hợp: trung bình 10 camera sau khi nối; EPFL: track mỗi frame và số ID sau khi nối, A / B):

  | Ngưỡng | MOTA | IDF1 | EPFL track/frame | EPFL số ID |
  |---|---|---|---|---|
  | 0 (tắt) | 0,723 | 0,822 | 4,81 / 4,63 | 199 / 191 |
  | 0,4 | 0,723 | 0,822 | 3,92 / 3,78 | 112 / 98 |
  | 0,5 | 0,723 | 0,822 | 3,77 / 3,51 | 95 / 81 |
  | **0,6** | **0,723** | **0,822** | **3,61 / 3,26** | **85 / 66** |
  | 0,7 | 0,716 | 0,818 | 3,46 / 3,14 | 74 / 60 |

  Chọn 0,6: mức cao nhất mà điểm tổng hợp chưa giảm. Nhận dạng lại trên tổng hợp không đổi ở mọi mức.
- **Kiểm tra không bỏ sót:** tỉ lệ khung đủ cao một người mà không có track nào phủ (IoU > 0,3) **giảm** 23% → 16% (A) và 19% → 11% (B). Luật mới không làm mất người được tìm thấy nguyên vẹn; nó còn giúp, vì track mảnh không còn chiếm chỗ.
- **Còn lại:** mỗi frame EPFL chỉ có khoảng 2,3 khung đủ cao, trong khi có tới 4 người. Người mặc đồ tối trước nền tối, và người đứng yên bị MOG2 dần coi là nền, không được tìm thấy. Đó là giới hạn của bước tìm người, không phải của tracker.

## QĐ-28: MOG2 quên nền chậm hơn (learning rate 0,005 → 0,001)
- **Trạng thái:** đã làm (2026-11-22), `detection.mog2_learning_rate: 0.001`.
- **Vấn đề:** MOG2 cập nhật nền ở **mọi** pixel, kể cả chỗ có người (không cập nhật chọn lọc được). Với 0,005, nền theo kịp sau khoảng 200 frame (8 giây ở 25 fps), nên người đứng yên vài giây bị "nuốt" vào nền rồi biến mất. Trên EPFL, ảnh debug có nhiều người đứng yên không có khung.
- **Hiệu chỉnh tham số** (tổng hợp: trung bình 10 camera sau khi nối; EPFL: số ID sau khi nối A / B):

  | learning rate | MOTA | IDF1 | Nhận dạng lại s4 (ngoại hình) | EPFL số ID |
  |---|---|---|---|---|
  | 0,005 | 0,723 | 0,822 | 2/4 | 85 / 66 |
  | 0,002 | 0,747 | 0,837 | 4/4 | 62 / 48 |
  | **0,001** | **0,755** | **0,836** | **4/4** | **54 / 48** |
  | 0,0005 | 0,760 | 0,846 | 4/4 | 41 / 55 |

  Chỉ đo bước tìm người trên EPFL, số khung đủ cao một người mỗi frame tăng 2,47 → 2,64 (A) và 2,25 → 2,57 (B).
- **Vì sao chọn 0,001 mà không chọn 0,0005:**
  - Video tổng hợp không có ánh sáng đổi, nên luôn có lợi cho việc quên thật chậm.
  - 0,0005 cần khoảng 80 giây mới theo kịp khi đèn đổi hoặc khi người đứng lâu rời đi (để lại "bóng ma"); 0,001 cần khoảng 40 giây.
  - Trên EPFL hai mức ngang nhau (A tốt hơn ở 0,0005, B tốt hơn ở 0,001).
- **Đã thử và bỏ:** trừ nền trung vị có cập nhật chọn lọc trên EPFL: 14–20% ảnh bị coi là foreground (báo nhầm nhiều).
- **Rủi ro:** khi ánh sáng thay đổi nhanh (bật đèn, mây che nắng qua cửa sổ), nền theo không kịp và báo nhầm cả mảng lớn trong vài chục giây.

## QĐ-29: Áp dụng kết quả hai vòng hiệu chỉnh tham số *(chấm chủ yếu trên video máy vẽ; phải làm lại trên video thật, QĐ-31)*
- **Trạng thái:** đã làm (2026-11-22), bằng `tools/hieu_chinh.py`. Chi tiết từng lượt ở [14](14_hieu_chinh_tham_so.md).
- **Cách làm:** mỗi vòng đổi riêng từng tham số (2 giá trị quanh giá trị đang dùng). Một giá trị chỉ được giữ khi **cả hai** điều kiện đúng:
  - điểm trên 5 cảnh tổng hợp, `(MOTA + IDF1) / 2`, không giảm, và nhận dạng lại không giảm;
  - trên EPFL, số ID không tăng và số track mỗi frame không tụt mạnh.

  Sau đó chấm lại bản gộp, vì các tham số ảnh hưởng lẫn nhau.
- **Thay đổi đã áp dụng:**

  | Tham số | Cũ | Mới |
  |---|---|---|
  | `detection.min_area` | 150 | 300 |
  | `people_tracker.iou_gate` | 0,1 | 0,2 |
  | `people_tracker.max_misses` | 10 | 3 |
  | `people_tracker.new_track_min_height_ratio` | 0,6 | 0,7 |
  | `tracklet_linking.min_color_similarity` | 0,5 | 0,3 |
  | `tracklet_linking.max_cost` | 1,0 | 1,3 |
- **Kết quả:**

  | | Điểm tổng hợp | MOTA | IDF1 | Nhận dạng lại | EPFL số ID A / B | EPFL track ma A / B |
  |---|---|---|---|---|---|---|
  | Trước | 0,796 | 0,755 | 0,836 | 18/20 \| 18/20 | 54 / 48 | 12% / 13% |
  | Sau | **0,825** | **0,779** | **0,870** | **20/20 \| 20/20** | **34 / 31** | **7% / 8%** |

  "Track ma" là track không đứng trên khung tìm được nào (trôi ở chỗ không có ai). Tỉ lệ người nguyên vẹn được track bám giữ nguyên (89% / 92%), nên phần giảm số track là bớt track thừa, không phải bỏ sót thêm người.
- **Đã thử nhưng không giữ** (điểm tổng hợp tăng nhưng nhận dạng lại giảm 20/20 → 18/20): `mog2_var_threshold` 30, `merge_max_frames` 60, `kalman.acceleration_std_px` 0,2.
- **Để ngỏ** (tổng hợp không đổi, EPFL ít ID hơn, nhưng chưa kiểm chứng được là đúng hay chỉ là gộp nhầm): `mog2_learning_rate` 0,0005, `confirm_hits` 4, `tracklet_linking.max_cost` 1,6.
- **Rủi ro:** chỉnh trên 5 cảnh tổng hợp và một video thật; với video khác (ánh sáng, góc quay khác) có thể phải chỉnh lại.

## QĐ-30: Ổn định hình trước khi tìm người; chỉ dùng mô hình chiều cao khi nó khớp
- **Trạng thái:** đã làm (2026-11-22), `src/detect/stabilization.py`, `src/detect/height_model.py`.
- **Bối cảnh:** chạy hệ thống trên **mọi** video thật trong `data/real`, không chỉ EPFL c0+c1:
  - `dev/S1` (giếng trời nhiều tầng, 1024×576, 18 giây, một camera): 48 ID, khoảng 10 khung mỗi frame. So với frame đầu, khung hình **trôi dần tới khoảng 40 px** (đo bằng tương quan pha), dù giữa hai frame liền nhau chỉ trôi khoảng 0,15 px. Mép cửa sổ, cột, lan can bị coi là "đang chuyển động".
  - Mô hình chiều cao của S1 không khớp: người ở nhiều tầng nên "chân càng thấp thì người càng cao" sai. Mô hình đoán người cao 95–111 px trong khi người thật cao khoảng 52 px, nên khung người thật bị chặn, không được tạo track (QĐ-27).
- **Quyết định:**
  1. **Ổn định hình** (`stabilization`): bám các điểm góc của frame đầu sang frame hiện tại bằng Lucas-Kanade, ước lượng phép dời + xoay + co giãn đều bằng RANSAC (`cv2.estimateAffinePartial2D`; điểm trên người đang đi bị loại vì không theo chuyển động chung), rồi đưa frame về tọa độ frame đầu. Góc ảnh dời dưới 1 px thì giữ nguyên frame. Box ghi ra file được đổi ngược về tọa độ video gốc.
  2. **Kiểm tra mô hình chiều cao:** độ lệch tương đối trung vị của các điểm so với đường thẳng > 15% (`height_model.max_residual`) thì không dùng mô hình. Đo được: video hợp giả định 1–11%, S1 22%.
- **Kết quả** (`outputs/hieu_chinh/05_on_dinh_hinh/`):

  | | Không ổn định hình | Có ổn định hình |
  |---|---|---|
  | S1: foreground (frame 300) | 9,8% ảnh | 1,3% ảnh |
  | S1: số ID / track mỗi frame | 66 / 11,7 | **22 / 3,1** |
  | 5 cảnh tổng hợp | 0,825, 20/20 | 0,825, 20/20 (không đổi) |
  | EPFL c0+c1, c2+c3 (camera cố định) | 34/31, 63/89 ID | 34/31, 62/89 ID (không đổi) |
- **Không phải học máy:** Lucas-Kanade, RANSAC và phép biến đổi affine đều là xử lý ảnh / ước lượng cổ điển.
- **Ghi chú (QĐ-32):** video S1 dùng để đo ở trên đã bị xóa theo yêu cầu của nhóm. Hiện không còn video thật nào bị trôi để kiểm tra lại.
- **Còn lại ở S1 (lúc đo):** phản chiếu trên kính cửa sổ vẫn sinh vài khung nhầm. Ổn định hình chỉ bù được camera trôi chậm hoặc rung nhẹ; camera quay theo người (lia máy) thì nền thay đổi hẳn, cách trừ nền không dùng được.

## QĐ-31: Chỉ dùng video thật; xóa toàn bộ video và hình máy vẽ *(phần dữ liệu EPFL tải thêm đã bỏ theo QĐ-32)*
- **Trạng thái:** đã làm (2026-11-23). Đây là **quy tắc bắt buộc** của nhóm.
- **Quyết định:**
  - Không dùng ảnh hay video do máy vẽ cho **bất cứ việc gì**: chấm điểm, hiệu chỉnh tham số, test, minh họa, báo cáo.
  - Đã xóa video máy vẽ (`data/synthetic`, `data/synthetic_v2`), 5 công cụ sinh video, các test và tài liệu dựa trên chúng (`docs/01_video_tong_hop.md`). Các file này vẫn còn trong lịch sử git nếu cần tra lại.
  - **Test có hình ảnh** chạy trên video thật (EPFL, S1). **Test chỉ có con số** (tọa độ khung, ma trận chi phí, công thức) giữ nguyên, vì không phải hình ảnh. Máy thiếu dữ liệu thì test cần video tự bỏ qua kèm lời nhắc.
- **Dữ liệu:** lúc đầu nhóm tải thêm hai bộ EPFL có đáp án (phòng thí nghiệm 6 người, lối đi ngầm) để chấm điểm.
  Theo QĐ-32, hai bộ này và video S1 đã bị xóa; dự án chỉ dùng video do nhóm đưa. Các con số đo trên hai bộ đó
  không còn dùng.
- **Phát hiện vẫn đúng với video của nhóm** (cùng cỡ ảnh 360×288):
  - Hệ thống trên video thật yếu hơn nhiều so với con số trên video máy vẽ trước đây.
  - Điểm đặc trưng SIFT khớp ít, màu áo phân biệt kém khi người cùng tông màu.
  - Các ngưỡng chọn trên video máy vẽ (`reid.match_threshold`, `tracklet_linking.min_color_similarity`…) phải hiệu chỉnh lại trên video thật có đáp án.
- **Hệ quả:** các quyết định hiệu chỉnh trước đây (QĐ-25 … QĐ-29) có phần dựa trên điểm video máy vẽ, nên phải
  hiệu chỉnh lại khi video của nhóm có đáp án.

## QĐ-32: Chỉ dùng video do nhóm đưa
- **Trạng thái:** đã làm (2026-11-23), theo yêu cầu của nhóm.
- **Quyết định:**
  - Dự án chỉ dùng video thật **do nhóm đưa**. Hiện là bộ EPFL phòng thí nghiệm 4 người: 4 file avi gốc, 4 camera, mỗi file 3915 frame.
  - Đã xóa:
    - hai bộ EPFL tải thêm (phòng thí nghiệm 6 người, lối đi ngầm), kèm đáp án, file hiệu chỉnh và các cảnh của chúng;
    - video S1;
    - bản mp4 cũ của bộ 4 người, vì đã thay bằng bản avi gốc nhóm đưa;
    - công cụ đổi đáp án EPFL (`tools/epfl_dap_an.py`).
  - Không tự tải thêm video hay bộ dữ liệu nào nếu nhóm không yêu cầu.
- **Hệ quả:**
  - **Chưa có video nào có đáp án**, nên chưa tính được MOTA, IDF1 hay số người ghép đúng; hệ thống chỉ đếm được số ID và số track ([12](12_he_thong.md)).
  - Test tự động chạy trên 4 video của nhóm và kiểm tra những thứ không cần đáp án ([15](15_du_lieu_that_va_dap_an.md)):
    - khung đúng cỡ người theo hiệu chỉnh camera (trung vị 0,90);
    - số tên thừa (7 tên trong 1000 frame đầu);
    - màu áo cùng người giống hơn khác người;
    - …
  - Chế độ oracle của nhận dạng lại (`tools/run_reid.py`, `tools/explain_reid.py`) cần đáp án nên tạm chưa dùng được.
  - Không còn video thật nào có camera bị trôi, nên bước ổn định hình (QĐ-30) chỉ còn được kiểm tra bằng con số.
- **Bước tiếp theo:** nhóm khoanh đáp án cho một đoạn video bằng `tools/annotate.py`, rồi hiệu chỉnh lại tham số theo điểm thật.
