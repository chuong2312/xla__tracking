# Hỏi đáp chuẩn bị bảo vệ

> Đây là bản khởi đầu. Nội dung lấy từ XLA mục 5 và mục 2.13, cộng thêm các câu về thiết kế mới trong [de_xuat_phuong_an.md](de_xuat_phuong_an.md). Mỗi module làm xong sẽ bổ sung câu hỏi vào đây.
> **Cách ôn:** che phần trả lời, tự nói câu trả lời thành lời, rồi so lại. Câu nào chưa nói trôi chảy thì đánh dấu để ôn tiếp.
> Các câu thuộc phần chưa cài (đánh dấu *chưa cài*) chỉ dùng được sau khi module đó xong.

## Ràng buộc "không ML"

**Đồ án có dùng học máy không?**
Đồ án không dùng mô hình học máy nào được huấn luyện: không YOLO, không mạng nơ-ron, không bộ phân loại. Mỗi bước thuộc một trong bốn loại:
- Xử lý ảnh: trừ nền, morphology, connected components, histogram.
- Ước lượng: Kalman, Lucas-Kanade.
- Tối ưu tổ hợp: Hungarian.
- Đặc trưng thiết kế sẵn: SIFT, ORB.

Không có tập huấn luyện, và không có mô hình nào được "fit" từ dữ liệu. Riêng MOG2 thì cần trả lời cẩn thận hơn (xem câu tiếp theo).

**MOG2 có phải học máy không?**
Không nên trả lời đơn giản là "có" hay "không". Nên trả lời theo các ý sau:
- MOG2 là mô hình nền hỗn hợp Gaussian thích nghi. Với mỗi pixel, nó giữ vài phân bố Gaussian mô tả những giá trị nền hay gặp, và cập nhật dần các phân bố đó qua từng frame.
- MOG2 chỉ phát hiện **foreground** (vùng khác nền), không biết pixel nào là người. Việc xác định người nằm ở các bước sau: lọc hình dạng rồi tracking.
- Việc cập nhật là ước lượng thống kê trực tuyến theo công thức cố định, không có nhãn và không có tập huấn luyện. Có tài liệu xếp nó vào học không giám sát, nên đây là vùng xám.
- Thầy đã cho dùng (2026-11-21). Nhóm vẫn cài thêm trừ nền trung vị để so sánh, và chọn cách mặc định theo kết quả trên dev.

**Chọn trọng số w1, w2, w3 có phải là "train" không?**
Không. Nhóm quét lưới vài giá trị rồi chọn theo chỉ số trên tập hiệu chỉnh (dev), giống như chọn ngưỡng Canny. Không có mô hình nào được fit. Tập test được giữ riêng chỉ để đánh giá.

**`knnMatch` có phải thuật toán KNN không?**
Không. Hàm này chỉ tìm 2 descriptor gần nhất để áp dụng tỉ lệ Lowe. Không có huấn luyện và không phân lớp.

**ORB và FAST có dùng học máy không?**
Không.
- Tiêu chí góc của FAST là một phép thử đoạn (segment test) cố định. Cây quyết định trong bài báo gốc chỉ dùng để sắp thứ tự kiểm tra cho nhanh, và mô phỏng đúng tiêu chí đó.
- Các cặp điểm BRIEF của ORB do tác giả chọn một lần rồi cố định.

Đồ án không học gì từ dữ liệu.

**Kalman có phải học máy không?**
Không. Kalman là bộ ước lượng tuyến tính: nó kết hợp dự đoán theo mô hình chuyển động với phép đo, và trọng số phụ thuộc vào độ nhiễu của từng nguồn. Các ma trận `F`, `Q`, `R` do nhóm đặt trong config, không học từ dữ liệu. Thầy đã cho dùng Kalman; nhóm dùng `cv2.KalmanFilter`.

**Dùng hàm có sẵn của OpenCV thì làm sao chứng minh là hiểu?**
Thầy cho dùng thư viện. Nhóm chứng minh hiểu bằng ba cách:
- Tự đặt mọi tham số và ma trận, ví dụ `F`, `H`, `Q`, `R` của Kalman, hay ngưỡng ratio test.
- Giải thích được công thức bên trong từng hàm.
- Mỗi module có ảnh minh họa từng bước (`--explain`), cho thấy dữ liệu đi qua thuật toán như thế nào.

## Phát hiện (C)

**Trừ nền trung vị hoạt động thế nào?** *(chưa cài)*
- Với mỗi pixel, lấy trung vị theo thời gian trên vài giây cảnh trống. Kết quả là ảnh nền.
- Pixel nào khác nền quá ngưỡng T thì là foreground.
- Trung vị bền với nhiễu hơn trung bình: một người đi qua trong vài frame không làm lệch trung vị.

**Vì sao phải cập nhật nền có chọn lọc?** *(chưa cài)*
Ánh sáng thay đổi dần nên nền phải được cập nhật. Nhưng nếu cập nhật cả ảnh thì người đứng yên sẽ dần bị "nuốt" vào nền và biến mất. Vì vậy chỉ cập nhật ở những pixel **không thuộc** track đã xác nhận.

**Bóng người được xử lý thế nào?** *(chưa cài)*
Bóng làm pixel tối đi nhưng giữ gần nguyên màu. Trong không gian HSV, một pixel là bóng nếu thỏa cả hai điều kiện:
- Tỉ lệ độ sáng `V_frame/V_nền` nằm trong khoảng `[α, β]`.
- Chênh lệch kênh H và kênh S nhỏ.

Pixel bóng bị bỏ khỏi foreground (Cucchiara 2003). Phản chiếu trên kính không phải bóng theo nghĩa này, nên được xử lý bằng mask ROI và lọc theo hình dạng.

**Vì sao kernel closing là 3×9?** *(chưa cài)*
Kernel thiên dọc nối được thân với chân khi chúng bị đứt (do quần áo trùng màu nền), mà ít nối ngang hai người đứng cạnh nhau. Có ablation so với kernel vuông.

## Bám người (D)

**Vì sao dùng Lucas-Kanade mà không dùng Farneback?** (XLA 2.13, câu trả lời đã sửa)
- LK là sparse: chỉ tính flow tại các điểm góc tốt, nơi bài toán có điều kiện tốt. Vì vậy LK rẻ và đủ để dịch box.
- Farneback tính flow cho mọi pixel nên tốn hơn nhiều. Nếu muốn kiểm tra ngược (forward-backward) trên dense flow thì chi phí còn tăng gấp đôi.
- Phần lớn pixel trong box là vùng trơn, nơi flow kém tin cậy do vấn đề khẩu độ (aperture problem).

**Lưu ý:** không được nói "Farneback không kiểm tra ngược được", vì dense flow vẫn kiểm tra ngược được.

**Forward-backward check là gì?**
Chạy LK từ frame t sang frame t+1, rồi từ vị trí mới chạy ngược về frame t. Điểm tốt phải quay về gần chỗ cũ. Điểm nào lệch quá `fb_thresh` pixel thì bị bỏ.

**Vì sao lấy trung vị độ dịch mà không lấy trung bình?**
Trung vị bền với điểm sai (outlier). Vài điểm bám nhầm sang nền hoặc sang người khác không kéo lệch được kết quả.

**Vì sao chỉ lấy điểm LK trong vùng box ∩ foreground?** *(chưa cài)*
Box hình chữ nhật chứa nhiều nền, và nền thường có nhiều góc hơn quần áo trơn. Điểm nền có độ dịch bằng 0, nên sẽ kéo trung vị về 0 và làm box "dính" vào nền khi người đi tiếp (XLA 2.1).

**Hungarian giải bài toán gì? Vì sao không ghép tham lam?**
Hungarian ghép một-một giữa track và detection sao cho tổng chi phí nhỏ nhất, với độ phức tạp O(n³). Ghép tham lam chọn cặp tốt nhất trước, nên có thể ép các cặp sau phải ghép sai. Hungarian thì tối ưu toàn cục.

**Vì sao cặp bị cấm dùng chi phí 1e6 mà không dùng vô cùng?**
`linear_sum_assignment` báo lỗi "infeasible" nếu có một hàng toàn giá trị vô cùng, và hàm này luôn gán đủ cặp. Vì vậy phải dùng một số hữu hạn lớn, rồi sau khi gán thì lọc bỏ các cặp vượt ngưỡng (XLA 2.9).

**Hai người chồng lên nhau thì xử lý thế nào?** *(chưa cài)*
- Phát hiện blob gộp **trước** khi ghép: đó là một detection chứa từ 2 box dự đoán trở lên.
- Các track liên quan chuyển sang `merged`. Ở trạng thái này, box không cập nhật theo blob, không lấy thêm điểm LK, và vị trí lấy theo Kalman.
- Khi hai người tách ra, ghép lại bằng ba tín hiệu: tính liên tục của chuyển động, so khớp ảnh mẫu (NCC), và màu.

**Người đứng yên thì sao?** *(chưa cài)*
Có hai lớp bảo vệ:
1. Nền chỉ cập nhật có chọn lọc, nên người đứng yên không bị nuốt vào nền.
2. Nếu vẫn mất detection mà điểm LK còn tin cậy và gần như không dịch, track chuyển sang `static` và không bị tính là mất.

## Đặc trưng và định danh (F, G)

**Vì sao không tô đen nền rồi mới trích keypoint?** *(chưa cài)*
Tô đen nền tạo ra cạnh tương phản nhân tạo dọc viền người, và detector sẽ bắt keypoint giả ở đó (đo được 26% keypoint nằm ở viền). Viền của hai người khác nhau lại trông giống nhau, nên sinh ra khớp giả. Cách đúng là dùng tham số `mask` của detector, với mask đã được erode (XLA 2.6).

**Vì sao không dùng màu làm đặc trưng chính?**
Màu lệch giữa hai camera do cân bằng trắng khác nhau, và kịch bản S4 có hai áo cùng màu. Vì vậy màu chỉ là đặc trưng bổ trợ, và được chuẩn hóa gray-world riêng cho từng camera.

**ASIFT khác SIFT thế nào? `sift_auto` là gì?** *(chưa cài)*
- **SIFT:** tìm keypoint bất biến với tỉ lệ (to/nhỏ) và với phép xoay, nhưng kém khi góc nhìn nghiêng nhiều.
- **ASIFT** (Morel và Yu, 2009): mô phỏng nhiều phiên bản ảnh như được nhìn từ các góc nghiêng khác nhau, rồi chạy SIFT trên từng bản. Nhờ vậy ASIFT vẫn khớp được khi hai camera lệch góc lớn.
  - Cái giá phải trả: chậm hơn và sinh nhiều keypoint hơn. Đo trên crop 128×256: ASIFT mất 64 ms, ra 8.206 keypoint; SIFT mất 19 ms, ra 501 keypoint.
  - Trong OpenCV, ASIFT là `cv2.AffineFeature_create`. Không phải phiên bản nào cũng có hàm này; nhóm đã chạy thử thành công trên bản 5.1.0.
- **`sift_auto`:** vẫn là SIFT thường, nhưng tự hạ ngưỡng khi crop có quá ít keypoint.

Thầy không nói rõ "auto SIFT" là nghĩa nào, nên nhóm làm cả hai và so sánh với cùng số keypoint.

**Vì sao không dùng RANSAC làm mặc định?** *(chưa cài)*
`estimateAffinePartial2D` giả định hai tập điểm có quan hệ đồng dạng. Giả định này đúng với vật phẳng và cứng, nhưng không đúng với cơ thể người nhìn từ hai góc (tay vung, chân bước). Mặc định nhóm dùng ràng buộc dọc đơn giản hơn: đầu phải khớp với đầu. RANSAC có trong ablation, và kết quả so sánh được báo cáo (XLA 2.8).

**Homography mặt sàn là gì? Nó có luôn ghép đúng không?** *(chưa cài)*
Homography là ma trận 3×3 ánh xạ một điểm trên mặt sàn từ ảnh camera A sang ảnh camera B. Ma trận này tính từ ít nhất 4 điểm mốc click tay. Nhóm chiếu điểm chân người từ A sang B rồi so vị trí.

Đây chỉ là **thông tin hình học bổ trợ**, không phải lúc nào cũng đúng. Nó dựa trên giả định mặt sàn gần phẳng, và có thể sai khi:
- sàn không phẳng (có bậc, có dốc),
- chân bị che, hoặc box lệch khỏi chân thật,
- hai video đồng bộ lệch,
- hai người đứng sát nhau.

Vì vậy nó chỉ là một thành phần có trọng số trong điểm tương đồng, và nhóm báo cáo riêng kết quả "chỉ dùng ngoại hình".

**Vì sao số cặp đúng có thể bị "nhảy bậc"?**
Hiện tượng này chỉ xảy ra trong một điều kiện: **cả hai camera đều phát hiện đủ 4 người, và ghép một-một trọn vẹn**.
- Khi đó kết quả ghép là một hoán vị của 4 người.
- Một hoán vị không thể có đúng 3 điểm cố định: nếu 3 người đã ghép đúng thì người thứ 4 cũng phải đúng.
- Vì vậy số cặp đúng chỉ có thể là 0, 1, 2 hoặc 4.

Nếu thiếu track, hoặc có ngưỡng "không khớp ai", thì điều kiện trên không còn đúng. Nhóm báo cáo thêm toàn bộ ma trận tương đồng và rank-1 của từng truy vấn (XLA 3.2).

## Đánh giá (H)

**MOTA và IDF1 khác nhau thế nào?** *(chưa cài)*
- `MOTA = 1 − (bỏ sót + báo nhầm + đổi ID) / tổng số GT`. Chỉ số này nặng về khả năng phát hiện.
- IDF1 là điểm F1 của các phát hiện mang đúng ID, sau khi ghép một-một mỗi người với một track. Chỉ số này nặng về khả năng giữ danh tính.

**Đo bộ nhớ thế nào? Vì sao không dùng tracemalloc?**
`tracemalloc` chỉ thấy bộ nhớ cấp phát qua Python, không thấy bộ nhớ C++ của OpenCV. Đo thử thì `tracemalloc` báo 11,5 MB trong khi RSS thực tế khoảng 2 GB. Vì vậy nhóm đo RSS của tiến trình, và mỗi cấu hình chạy trong một tiến trình riêng (XLA 2.14).

**Tách dev/test thế nào? Có hạn chế gì?**
Mỗi kịch bản quay 2 lượt: một lượt để hiệu chỉnh tham số (dev), một lượt chỉ dùng để đánh giá (test). Hạn chế là cả hai lượt dùng cùng 4 người, nên kết quả Re-ID có thể lạc quan.
