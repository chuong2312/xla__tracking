# Đề xuất phương án: Theo dõi nhiều người và Định danh đa camera

> **Trạng thái:** đề xuất, ngày 2026-11-21; cùng ngày đã cập nhật theo câu trả lời của thầy (mục 3). Code hiện tại (tracker LK cũ, detector MOG2) **chưa** theo kiến trúc này. Việc viết lại bắt đầu từ giai đoạn 2 (mục 7).
> **Căn cứ:** mọi lựa chọn bám theo `XLA.md` (bản review roadmap). Ký hiệu *XLA 2.4* nghĩa là mục 2.4 trong file đó.
> **Tài liệu đi kèm:** [cau_hoi_thay.md](cau_hoi_thay.md) · [ke_hoach_quay_video.md](ke_hoach_quay_video.md) · [quyet_dinh_thiet_ke.md](quyet_dinh_thiet_ke.md) · [hoi_dap_bao_ve.md](hoi_dap_bao_ve.md)

## 1. Đồ án làm gì (đọc phần này trước)

**Đầu vào:** video từ 2 camera đặt cố định trên chân máy. Hai camera cùng nhìn một khu vực từ hai góc khác nhau (chồng vùng nhìn), trong cảnh có 4 người đi lại và một cánh cửa.

**Đầu ra:**
1. **ID cục bộ:** trong mỗi camera, mỗi người có một ID giữ nguyên suốt thời gian xuất hiện (`tracks.csv`).
2. **ID toàn cục:** nối track ở cam A với đúng người đó ở cam B (`global_ids.csv`).
3. **Sự kiện qua cửa:** danh sách các lượt vào và ra (`events.csv`).
4. **Bảng số liệu đánh giá trung thực:** đúng bao nhiêu, sai ở đâu, vì sao sai.

**Pipeline.** Mỗi camera chạy lần lượt C → D → E → F, sau đó bước G ghép kết quả của hai camera:

```
video ─► [C] Tìm vùng chuyển động ─► box ─► [D] Bám người ─► track ─► [E] Đếm qua cửa ─► events.csv
              │ foreground mask                               │
              └─────────────────────────────►[F] Đặc trưng ◄──┘
                                                   │ gallery của mỗi track
                        cam A ─┐                   ▼
                               ├─► [G] Ma trận tương đồng A×B ─► Hungarian ─► global_ids.csv
                        cam B ─┘
[H] Đánh giá mọi bước bằng ground truth gán tay
```

| Bước | Ý tưởng | Vì sao không phải ML |
|---|---|---|
| C | Ước lượng ảnh nền (trung vị của vài giây cảnh trống, hoặc MOG2). Pixel nào khác nền quá ngưỡng là foreground, tức vùng chuyển động. Lọc theo hình dạng xong mới ra các box ứng viên là người. | Ước lượng thống kê về nền theo công thức cố định, không có nhãn hay tập huấn luyện. Thầy đã cho dùng MOG2. |
| D | Kalman đoán vị trí tiếp theo theo vận tốc. Lucas-Kanade đo chuyển động thật từ ảnh. Hungarian ghép box đoán với box phát hiện. | Đây là ước lượng (Kalman; LK giải bình phương tối thiểu) và tối ưu tổ hợp (Hungarian). |
| E | Điểm chân của người đi qua hẳn vạch cửa, vượt cả vùng đệm, thì ghi một sự kiện IN hoặc OUT. | Hình học thuần. |
| F | Mỗi track giữ vài ảnh crop đẹp nhất. Từ đó trích keypoint SIFT/ORB và histogram màu. | Đặc trưng thủ công, có thiết kế cố định. |
| G | Độ giống nhau tính từ keypoint khớp và màu, cộng thêm thông tin bổ trợ là vị trí trên mặt sàn qua homography. Hungarian ghép người ở A với người ở B. | So khoảng cách rồi tối ưu tổ hợp; trọng số chọn bằng quét lưới. |

**Câu cần thuộc:** *"Tham số được hiệu chỉnh bằng quét lưới trên tập dev; không có mô hình nào được huấn luyện."* Không bao giờ dùng chữ "train" hay "huấn luyện" (XLA 1).

## 2. Vì sao đổi phương án

| Vấn đề hiện tại | Bằng chứng | Hướng xử lý |
|---|---|---|
| Video thật quay cầm tay | Đo bằng `phaseCorrelate`: camera trôi khoảng 25 px (ở độ rộng 640) sau 15 giây và có cả xoay. Mask MOG2 sáng toàn bộ các cạnh tường (`outputs/real_dev_S1/base/diag/`). | Quay lại bằng chân máy (QĐ-01). |
| Detector sinh quá nhiều box trên video thật | Có frame ra 13 detection. Video 18 giây chỉ có vài người mà sinh 134 track ID (55 track được xác nhận). | Phần lớn là hệ quả của vấn đề trên. Thêm cách trừ nền mới ở mục 4.C. |
| Tracker quá phức tạp mà vẫn sai | `src/tracker.py` dài 777 dòng, khoảng 1/3 là diagnostics. `test_tracker_crossing` có 3 lần đổi ID dù dùng detection chuẩn. | Viết lại theo Kalman + LK, mỗi file dưới 200 dòng (QĐ-05). |
| Phần chính của đề tài chưa có | `features`, `reid`, `events` mới chỉ là section rỗng trong config. | Làm sớm một bản tối thiểu chạy được từ đầu đến cuối (QĐ-09). |
| Code viết bằng AI, khó giải thích | Nhiều lệnh dồn trên một dòng bằng `;`, logic ownership có nhiều nhánh. | Áp dụng quy tắc code và viết tài liệu cho từng module (mục 6). |

## 3. Quyết định đã chốt (2026-11-21)

**Thầy trả lời** (chi tiết trong [cau_hoi_thay.md](cau_hoi_thay.md)):

| Nội dung | Thầy trả lời |
|---|---|
| Dùng AI, khai báo dùng AI | Thoải mái |
| MOG2 | Được dùng |
| Auto SIFT | Được dùng |
| Dùng thư viện có sẵn (OpenCV, SciPy) | Được dùng |
| Kalman | Được dùng |
| Đếm người IN/OUT | **Bắt buộc** |
| Số camera | Tối đa 4 camera cố định |
| Homography mặt sàn | Được dùng |

**Nhóm chốt:**
- Quay lại toàn bộ dữ liệu bằng chân máy.
- **2 camera chồng vùng nhìn** (XLA 2.11). Code vẫn thiết kế để chạy được tới 4 camera.
- Mỗi cảnh có **4 người**.
- **Viết lại tracker**, không sửa tiếp bản cũ.
- Auto SIFT: làm **cả `sift_auto` lẫn ASIFT**, rồi so với SIFT thường.
- Ưu tiên số 1 là **hiểu và bảo vệ được**.

Lý do chi tiết của từng quyết định nằm trong [quyet_dinh_thiet_ke.md](quyet_dinh_thiet_ke.md).

## 4. Thiết kế từng phần

### A. Cấu hình và môi trường (XLA 2.15, 4.5)

- **Resize giữ tỉ lệ:** video 16:9 thu về `640×360`. Mọi ngưỡng tính bằng pixel đều ghi ở độ phân giải này. Nếu đổi độ phân giải thì quy đổi bằng hàm `scaled()`.
- **Config riêng cho từng camera:** `cameras.<tên>.scene`, thiết kế cho tối đa 4 camera (giới hạn của thầy). Nhóm dùng 2 camera là `cam_a` và `cam_b`. Mỗi camera có ROI, vùng che, vạch cửa, khoảng diện tích người, `frame_offset` (để đồng bộ) và ma trận homography của sàn.
- **Ghim phiên bản** `opencv-python` trong `requirements.txt` và ghi phiên bản vào báo cáo. Bản đang dùng là 5.1.0, đã chạy thử có ASIFT.

### B. Dữ liệu (XLA 3.1, 3.3, 3.5, 3.6, 3.7)

Xem [ke_hoach_quay_video.md](ke_hoach_quay_video.md). File đó gồm checklist quay, danh sách kịch bản, cách đồng bộ, cách lấy homography, quy ước gán nhãn và ước tính số giờ gán nhãn.

### C. Tìm vùng chuyển động (XLA 1, 1.1, 2.2, 2.16)

Interface `BackgroundModel` có 2 cài đặt, chọn bằng `detector.background`. Thầy cho dùng cả hai, nên nhóm **cài cả hai và chọn cái mặc định theo kết quả ablation trên dev**:
- **`median_running`**: thuần xử lý ảnh. Cập nhật nền có chọn lọc được, nên có lợi thế ở kịch bản S3 (người đứng yên).
- **`mog2`**: mô hình nền hỗn hợp Gaussian thích nghi của OpenCV.
  - Ưu điểm: có sẵn phát hiện bóng, và về lý thuyết chịu được nền dao động tốt hơn.
  - Nhược điểm: `learningRate` áp cho cả ảnh, nên không cập nhật nền có chọn lọc được (XLA 2.2).
  - Phần này tái dùng code đã có trong `src/detector.py`.

Cả hai chỉ phát hiện **foreground** (vùng khác nền), không biết pixel nào là người. Box ứng viên là người chỉ có sau bước lọc hình dạng, còn danh tính chỉ có sau bước tracking.

Các bước của `median_running`:
1. **Khởi tạo nền.** Lấy trung vị theo thời gian của từng pixel trên N frame đầu (3–5 giây cảnh trống): `B(x) = median{I_1(x), …, I_N(x)}`.
2. **Hiệu với nền.** Tính `D(x) = ‖I(x) − B(x)‖` trên 3 kênh màu. Pixel có `D(x) > T` là foreground.
3. **Khử bóng** (Cucchiara 2003). Bóng làm pixel tối đi nhưng giữ gần nguyên màu. Vì vậy một pixel foreground được coi là bóng nếu thỏa đồng thời ba điều kiện trong HSV:
   - `α ≤ V_I/V_B ≤ β`
   - `|S_I − S_B| ≤ τ_S`
   - `|H_I − H_B| ≤ τ_H`
4. **Che ROI và vùng bỏ qua ngay sau bước 3**, trước khi làm morphology. Nếu che sau thì phép `dilate` sẽ làm lan vào vùng đã che (XLA 2.16).
5. **Morphology.** Làm opening để xóa chấm nhiễu, rồi closing với kernel **3×9 thiên dọc**. Kernel dọc nối được thân với chân mà ít nối hai người đứng cạnh nhau.
6. **Connected components**, rồi lọc theo diện tích và tỉ lệ cao/rộng để ra các box.
7. **Cập nhật nền có chọn lọc.** Cập nhật `B ← (1−a)·B + a·I`, nhưng **chỉ tại pixel không thuộc track đã xác nhận**, bằng `cv2.accumulateWeighted(I, B, a, mask)`. Nhờ vậy người đứng yên không bị "nuốt" vào nền (XLA 2.2), trong khi thay đổi ánh sáng nhẹ vẫn được cập nhật vào nền.

**Đầu ra:**
- `detections.csv`, giữ đúng các cột cũ `frame,x,y,w,h,area`.
- Foreground mask, dùng cho D (chọn điểm LK) và F (mask khi trích keypoint).

**Xóa hẳn** HOG+SVM khỏi code (XLA 1).

### D. Bám người: tracker mới (XLA 2.1–2.5, 2.9)

**Mỗi track lưu:**
- Trạng thái Kalman `x = [cx, cy, vx, vy]`.
- Kích thước box `w, h`, làm mượt bằng trung bình trượt.
- Các điểm LK.
- Ảnh mẫu (template) để so khớp NCC.
- Histogram HSV.
- Trạng thái và các bộ đếm.

**Kalman vận tốc hằng**, dùng `cv2.KalmanFilter` (thầy cho dùng thư viện). Nhóm tự đặt các ma trận `F`, `H`, `Q`, `R`, nên vẫn giải thích được từng bước:
- **Dự đoán:** `x ← F·x` và `P ← F·P·Fᵀ + Q`. Ma trận `F` mô tả "vị trí mới = vị trí cũ + vận tốc".
- **Hiệu chỉnh bằng phép đo `z`:**
  - `K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹`
  - `x ← x + K·(z − H·x)`
  - `P ← (I − K·H)·P`
- **Hiểu đơn giản:** kết quả là **trung bình có trọng số** giữa dự đoán và phép đo, nguồn nào ít nhiễu hơn thì được tin nhiều hơn. Hai ma trận nhiễu `Q`, `R` do nhóm đặt trong config, không học từ dữ liệu.

**Các bước trong mỗi frame:**
1. **Dự đoán.** Chạy Kalman predict cho mọi track.
2. **Đo chuyển động bằng LK.**
   - Điểm của track lấy bằng `goodFeaturesToTrack`, chỉ trong vùng **box ∩ erode(foreground)** để không lấy nhầm điểm nền (XLA 2.1).
   - Chạy LK xuôi rồi chạy ngược lại (forward-backward). Điểm nào quay về lệch quá `fb_thresh` thì bỏ.
   - Nếu còn ít nhất `min_points` điểm: vị trí theo LK = tâm cũ + **trung vị** độ dịch. Vị trí này được dùng làm một phép đo cho Kalman, với nhiễu `R_lk`.
3. **Phân loại detection trước khi ghép** (XLA 2.3):
   - *Gộp:* một detection chứa từ 2 box dự đoán trở lên (độ phủ đạt ngưỡng và tâm box nằm trong detection). Các track đó chuyển sang `merged`, và detection này bị loại khỏi bước ghép.
   - *Tách:* có từ 2 detection cùng nằm trong một box dự đoán (người bị cắt ngang giữa thân và chân). Các detection đó được gộp lại thành 1 trước khi ghép.
   - *Sạch:* mọi detection còn lại.
4. **Ghép bằng Hungarian** giữa track và detection sạch, chi phí `1 − IoU`.
   - Cặp bị cấm nhận chi phí hữu hạn lớn `1e6`. Gán xong thì lọc bỏ các cặp vượt ngưỡng (XLA 2.9).
   - Hàm ghép này dùng chung với bước G, đặt trong `assignment.py`.
5. **Cập nhật track được ghép:**
   - Kalman correct bằng tâm detection, với nhiễu `R_det`.
   - Cập nhật `w, h`.
   - Định kỳ lấy lại điểm LK.
   - Cập nhật template và histogram, chỉ từ detection sạch.
6. **Track không được ghép:**
   - **Đang `merged`:**
     - Không lấy thêm điểm LK, vì sẽ hút nhầm điểm của người kia (XLA 2.5).
     - Bỏ các điểm nằm trong vùng chồng lấn của người đứng sau. Người có đáy box thấp hơn trong ảnh (y đáy lớn hơn) thường gần camera hơn, tức là đứng trước.
     - Vị trí lấy theo Kalman.
   - **Đứng yên:** nếu track còn điểm LK tin cậy và gần như không dịch chuyển thì chuyển sang `static` và **không** tăng `misses` (XLA 2.2).
   - **Còn lại:** `misses += 1`. Quá `max_misses` thì chuyển sang `lost`.
7. **Khi blob gộp tách ra.** Ghép lại các track của nhóm với các detection mới bằng một bài Hungarian nhỏ. Chi phí kết hợp ba tín hiệu (XLA 2.4):
   - Khoảng cách tới vị trí Kalman dự đoán (tính liên tục của chuyển động).
   - `1 − NCC` so với template chụp trước khi gộp.
   - Khoảng cách Bhattacharyya giữa histogram HSV.

   Không dựa riêng vào màu, vì kịch bản S4 có hai áo cùng màu.
8. **Detection sạch chưa được ghép** tạo track mới ở trạng thái `tentative`. Track chỉ được `confirmed` khi có đủ N hit liên tiếp **và** đã dịch chuyển một khoảng tối thiểu. Điều kiện thứ hai loại được phản chiếu trên kính (XLA 2.16).

**Máy trạng thái:**

| Trạng thái | Vào khi | Ra khi |
|---|---|---|
| `tentative` | Có detection sạch mới | Đủ N hit và đã dịch chuyển → `confirmed`. Mất trước khi đủ → xóa. |
| `confirmed` | Từ `tentative` | Nằm trong blob gộp → `merged`. Không có detection nhưng LK cho thấy đứng yên → `static`. Quá `max_misses` hoặc ra khỏi ảnh → `lost`. |
| `merged` | Nằm trong blob gộp | Blob tách ra và ghép lại được (bước 7) → `confirmed`. Quá `merge_max_frames` → `lost`. |
| `static` | Không có detection, LK tin cậy, độ dịch ≈ 0 | Có detection trở lại → `confirmed`. Quá `static_max_frames` → `lost`. |
| `lost` | Từ các trạng thái trên | Kết thúc. ID không bao giờ được dùng lại. |

**Đầu ra:** `tracks.csv`, giữ đúng các cột cũ `frame,track_id,state,x,y,w,h,n_points`.

### E. Đếm qua cửa (XLA 2.17, 3.5): bắt buộc

- Thầy yêu cầu phần này, nên dữ liệu quay phải có cảnh cửa.
- Dùng **điểm chân** của track: `p = (cx, y + h)`.
- Vạch cửa là một đoạn thẳng khai báo trong config. Điểm chân nằm phía nào của vạch được xác định bằng dấu của tích có hướng.
- Có **vùng đệm hysteresis** rộng `crossing_margin`. Một lượt qua chỉ được tính khi điểm chân ra khỏi vùng đệm ở phía bên kia. Nhờ vậy người đứng ngay trên vạch không bị đếm nhiều lần.
- Sự kiện được ghi **ngay khi qua vạch**, trước khi track có thể bị kết thúc.
- `events.csv` có các cột `event_id,camera,frame,track_id,global_id,direction`.
- **Test hồi quy bắt buộc:** một người vào, một người ra thì kết quả phải đúng `IN 1 / OUT 1`, và không được dựa vào màu.

### F. Đặc trưng ngoại hình (XLA 2.6, 2.7, 2.12, 2.18)

- **Gallery.** Mỗi track giữ K crop tốt nhất. Crop chỉ được chọn từ detection sạch, không ở trạng thái `merged`, không chạm mép ảnh, có diện tích lớn, và các crop cách nhau đủ xa về thời gian.
- **Chuẩn hóa crop** về 128×256, giữ tỉ lệ và đệm biên bằng `copyMakeBorder`.
- **Keypoint.** Gọi `detectAndCompute(gray, mask=erode(fg, k))`, với `k` xấp xỉ bán kính patch.
  - **Không** tô đen nền trên ảnh. Làm vậy sinh keypoint giả dọc viền người (XLA 2.6 đo được 26% keypoint nằm ở viền).
  - SIFT dùng `nfeatures` cố định.
  - ORB dùng `edgeThreshold=8, patchSize=15`. Với tham số mặc định, ORB chỉ lấy được keypoint ở dải giữa của crop nhỏ (XLA 2.7).
  - Khi so sánh ORB với SIFT, giữ số keypoint của hai bên tương đương nhau.
- **Màu.** Chuẩn hóa gray-world riêng cho từng camera, vì hai điện thoại cân bằng trắng khác nhau. Tính histogram HSV riêng cho **thân trên** và **thân dưới**.
- **"Auto SIFT":** thầy cho dùng nhưng không nói rõ là nghĩa nào. Nhóm làm cả hai nghĩa, rồi so với SIFT thường:
  - **SIFT thường:** `SIFT(nfeatures=N)`.
  - **`sift_auto`:** khi crop có quá ít keypoint thì hạ `contrastThreshold` rồi chạy lại (XLA 2.18). Ghi lại số lần phải chạy lại trung bình và thời gian chạy.
  - **ASIFT:** `cv2.AffineFeature_create(cv2.SIFT_create())`. Hàm này mô phỏng nhiều góc nghiêng, nhờ vậy vẫn khớp được khi hai camera nhìn lệch góc nhau.
    - Không phải phiên bản OpenCV nào cũng có hàm này. Nhóm đã chạy thử thành công trên OpenCV 5.1.0.
    - Trên crop 128×256: ASIFT ra 8.206 keypoint trong 64 ms, còn SIFT ra 501 keypoint trong 19 ms.
  - ASIFT sinh ra nhiều keypoint hơn hẳn. Để so sánh công bằng, cả ba nhánh đều chỉ giữ N keypoint mạnh nhất.

### G. Định danh đa camera (XLA 2.8, 2.10, 2.11, 3.2)

1. **Nối tracklet trong cùng một camera trước.** Hai track được coi là cùng một người nếu thỏa cả bốn điều kiện:
   - Không trùng thời gian.
   - Khoảng hở giữa hai track ngắn.
   - Ngoại hình giống nhau.
   - Vị trí hợp lý.

   Bước này giảm tình trạng một người bị tách thành nhiều track, ví dụ trong XLA 2.10 là "5 track cho 3 người".
2. **Điểm tương đồng giữa track `a` (cam A) và track `b` (cam B):**
   - **`S_feat`:**
     - Với từng cặp crop trong hai gallery, dùng `knnMatch(k=2)` kèm ratio test của Lowe.
     - Chỉ giữ cặp khớp **nhất quán theo chiều dọc**: `|y_a/H − y_b/H| < δ`, nghĩa là đầu phải khớp với đầu.
     - Nếu ít hơn 6–8 cặp khớp thì điểm bằng 0. Lấy giá trị tốt nhất trên các cặp crop.
     - RANSAC `estimateAffinePartial2D` chỉ dùng cho **ablation**, vì người không phải vật cứng (XLA 2.8).
   - **`S_color`:** bằng `1 −` khoảng cách Bhattacharyya, lấy trung bình của thân trên và thân dưới.
   - **`S_geo`:** chỉ là thông tin hình học **bổ trợ**, dựa trên giả định mặt sàn gần phẳng. Không được mặc định là nó luôn ghép đúng.
     - Xét các frame đã đồng bộ mà cả hai camera cùng thấy người đó.
     - Chiếu điểm chân của `a` sang ảnh B bằng homography sàn `H_AB`, rồi đo khoảng cách `d` tới điểm chân của `b`.
     - Tính `S_geo = exp(−d²/2σ²)`. Nếu không có frame chung thì bỏ thành phần này.
     - Các trường hợp `S_geo` có thể sai:
       - sàn không phẳng (có bậc, có dốc),
       - chân bị che hoặc box lệch khỏi chân thật,
       - hai video đồng bộ lệch,
       - hai người đứng sát nhau.
   - **Tổng hợp:** `S = w1·S_feat + w2·S_color + w3·S_geo`. Các trọng số chọn bằng quét lưới trên dev.
3. **Ghép.**
   - Chạy Hungarian trên `1 − S` cho từng cặp camera. Nhóm dùng 2 camera nên chỉ có cặp A–B.
   - Cặp nào có `S < τ` bị loại và nhận `global_id = −1` ("không khớp ai").
   - Giới hạn cần biết (XLA 3.2): nếu cả hai camera đều phát hiện đủ 4 người **và** ghép một-một trọn vẹn, thì số cặp đúng chỉ có thể là 0, 1, 2 hoặc 4, không bao giờ đúng 3. Khi có ngưỡng "không khớp ai" hoặc bị thiếu track, giới hạn này không còn.
4. **Đánh giá hai chế độ:**
   - *Oracle:* dùng box ground truth, để đo riêng năng lực Re-ID.
   - *End-to-end:* dùng track thật. Mỗi track được gán `gt_id` bằng bỏ phiếu đa số theo IoU.
5. **Báo cáo:**
   - Tách riêng kết quả "chỉ ngoại hình" (`w3 = 0`) và "ngoại hình + hình học".
   - Kèm **toàn bộ ma trận tương đồng** và rank-1 của từng truy vấn.
   - Ghi kết quả dạng `x/y cặp đúng`, có nêu mẫu số.
6. **Đầu ra:**
   - `global_ids.csv` với các cột `cam,track_id,global_id`.
   - `reid_scores.csv` với các cột `cam_a_track,cam_b_track,method,score`.

   Đây đúng là định dạng mà `src/metrics.py` hiện đã đọc được.

### H. Đánh giá và benchmark (XLA 2.14, 3.4)

- **Detection:** precision và recall ở IoU ≥ 0,5, **chỉ tính trên keyframe gán tay**. Bỏ qua frame bị che nặng (`occluded = 2`).
- **Tracking:** thêm MOTA, MOTP, IDF1 (tự cài, vì chỉ là phép đếm), bên cạnh các chỉ số đã có: số lần đổi ID, số đoạn đứt, độ thuần.
  - `MOTA = 1 − (FN + FP + IDSW) / GT`
  - `MOTP` = trung bình IoU của các cặp ghép đúng.
  - `IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)`
- **Tách lỗi tracker khỏi lỗi detector:** chạy tracker với detection ground truth (có thêm nhiễu).
- **Tốc độ và RAM:**
  - Thời gian từng bước đo bằng `Timer`.
  - RAM đo bằng RSS (`psutil`) hoặc `ru_maxrss`, **mỗi cấu hình chạy một tiến trình riêng**.
  - Không dùng `tracemalloc`, vì nó không thấy bộ nhớ C++ của OpenCV.
- **Các ablation dự kiến:**
  - `median_running` so với `mog2` (kết quả này quyết định cái nào làm mặc định).
  - Kernel 3×9 so với kernel vuông.
  - Có và không có LK (chỉ dùng Kalman).
  - Có và không có kiểm tra forward-backward.
  - Có và không có xử lý blob gộp.
  - SIFT so với ORB, và SIFT so với `sift_auto` so với ASIFT, luôn giữ cùng số keypoint.
  - Ràng buộc dọc so với RANSAC so với không ràng buộc.
  - Có và không có chuẩn hóa màu.
  - Có và không có `S_geo`.
- **Failure gallery:** ảnh các ca sai điển hình, kèm giải thích.

## 5. Cấu trúc code đích

```
src/
  io_video.py      đọc video, resize giữ tỉ lệ, đồng bộ bằng frame_offset
  config.py        config theo từng camera, hàm scaled()
  background.py    BackgroundModel: MedianRunning, MOG2
  detection.py     mask → morphology → connected components → box
  kalman.py        Kalman vận tốc hằng (cv2.KalmanFilter)
  optical_flow.py  LK + forward-backward + trung vị độ dịch
  assignment.py    Hungarian + gating (dùng chung cho D và G)
  tracker.py       máy trạng thái track
  door.py          DoorCounter
  features.py      crop chuẩn hóa, SIFT/ORB, histogram màu, gray-world
  gallery.py       chọn K crop tốt nhất cho mỗi track
  reid.py          ma trận tương đồng, homography, ghép
  metrics.py       (giữ phần cũ) + MOTA/MOTP/IDF1, gán track → gt_id
  viz.py           overlay, ảnh minh họa từng bước (--explain)
run.py             --stage detect|track|events|reid|all --scene <thư mục cảnh>
tools/             annotate (thêm cột occluded), verify_gt, make_synthetic v2, calibrate_floor, ablate, benchmark
docs/              tài liệu này và tài liệu của từng module
```

**Giữ lại và tái sử dụng:**
- `src/utils.py`: `iter_video`, `iou`, `iou_matrix`, `Timer`, `set_seed`.
- Toàn bộ `src/metrics.py` và định dạng các file CSV.
- Phần ROI, morphology và connected components trong `src/detector.py`.
- `tools/annotate.py` và `tools/verify_gt.py`.
- Phần manifest provenance trong `tools/trace_tracking.py`.
- `tests/test_tracker_crossing.py`, dùng làm test chấp nhận cho tracker mới.
- Tracker cũ được giữ lại trong git bằng một tag trước khi thay.

**Video tổng hợp v2** (XLA 2.4):
- Mỗi hộp có màu và texture khác biệt rõ.
- **Thêm một cặp hộp cùng màu**, mô phỏng S4.
- Một kịch bản người đứng yên (S3).
- Một kịch bản qua cửa.

## 6. Cách làm việc để hiểu được code

Nhóm viết code cùng AI, nên mọi phần đều phải **giải thích được**.

**Quy tắc code:**
- Mỗi file một khái niệm, dưới khoảng 200 dòng; hàm ngắn.
- Không dồn nhiều lệnh trên một dòng bằng `;`.
- Docstring tiếng Việt nói rõ **vì sao** làm vậy và ghi **công thức**.
- Mọi ngưỡng nằm trong config, có chú thích ý nghĩa và đơn vị.
- Phần ghi log và diagnostics tách khỏi code thuật toán.

**Mỗi module khi xong phải có:**
1. **Tài liệu** `docs/NN_<module>.md`, theo thứ tự: bài toán → thuật toán → công thức → tham số → khi nào hỏng → thầy có thể hỏi gì.
2. **Lệnh `--explain`** xuất ảnh minh họa từng bước. Ví dụ: frame → nền → hiệu → mask → box; các điểm LK bị loại; ma trận tương đồng; các cặp SIFT khớp. Ảnh này dùng luôn cho báo cáo.
3. **Câu hỏi mới** bổ sung vào [hoi_dap_bao_ve.md](hoi_dap_bao_ve.md).
4. **Buổi giải thích:** Claude giải thích lại module trong chat và hỏi nhóm 3–5 câu tự kiểm tra. Nhóm chưa trả lời được thì chưa sang module tiếp theo.

## 7. Lộ trình

Lộ trình không có hạn cố định. Mỗi cổng nghiệm thu có **giới hạn tối đa 5 ngày**; quá hạn mà chưa đạt thì ghi nhận là hạn chế rồi làm tiếp (XLA 4.2).

Thứ tự ưu tiên theo XLA 4.4: C (bản tối thiểu) → D → F (chỉ SIFT) → G → H. ORB, auto SIFT và các ablation phụ làm sau cùng.

| Giai đoạn | Việc | Cổng nghiệm thu |
|---|---|---|
| 0. Ngay | Hỏi thầy theo [cau_hoi_thay.md](cau_hoi_thay.md): **đã xong ngày 2026-11-21**. Còn lại: commit và tag tracker cũ, chuẩn bị quay. | Có câu trả lời của thầy (đạt) |
| 1. Dữ liệu | Quay theo [ke_hoach_quay_video.md](ke_hoach_quay_video.md), chuyển sang frame rate cố định, click điểm homography, gán GT. Song song: làm video tổng hợp v2. | Đủ GT cho dev và test; `verify_gt` không còn cảnh báo |
| 2. C | `background.py`, `detection.py`, ablation MOG2 | Đếm đúng số người ở ít nhất 80% số frame trên dev |
| 3. D | `kalman.py`, `optical_flow.py`, `assignment.py`, `tracker.py`, MOTA/IDF1 | Test giao cắt với detection chuẩn: 0 lần đổi ID. Có số IDF1 trên dev. |
| 4. E | `door.py` (bắt buộc) | Test IN 1 / OUT 1 đạt; khớp với số đếm tay trên video cửa |
| 5. F + G | SIFT + màu + homography; làm chế độ oracle trước, end-to-end sau | Có kết quả `x/y` cặp đúng kèm mẫu số và ma trận tương đồng |
| 6. H | ORB, `sift_auto` và ASIFT, ablation, đo tốc độ và RAM | Có bảng ablation và failure gallery |
| 7. I | Báo cáo, demo, tập bảo vệ | Tự trả lời hết [hoi_dap_bao_ve.md](hoi_dap_bao_ve.md) |

## 8. Rủi ro và phương án dự phòng

Các rủi ro kiểu "thầy không cho dùng AI / MOG2 / Kalman / thư viện có sẵn" không còn nữa, vì thầy đã cho phép tất cả (2026-11-21).

| Rủi ro | Phương án dự phòng |
|---|---|
| Nhóm không giải thích được phần code viết bằng AI | Thầy cho dùng AI, nhưng vẫn áp dụng đầy đủ mục 6: tài liệu cho từng module, lệnh `--explain`, các buổi tự kiểm tra. |
| Homography sai ở một số chỗ: sàn không phẳng, chân bị che, đồng bộ lệch, hai người đứng sát nhau | `S_geo` chỉ là một thành phần bổ trợ có trọng số. Kết quả chỉ dùng ngoại hình được báo cáo riêng, và các ca sai đưa vào failure gallery. |
| ASIFT chậm và sinh rất nhiều keypoint | Chỉ chạy ASIFT trên K crop của gallery, không chạy trên mọi frame. Giữ N keypoint mạnh nhất để so công bằng với SIFT. Đo thời gian riêng cho ASIFT. |
| 4 người trong cảnh che nhau nhiều | Ở S1 cho mọi người đi thưa, ít che nhau. Chỉ S2 và S4 mới cố tình cho đi giao cắt. |
| Sàn không có điểm mốc rõ | Dán ít nhất 4 điểm băng keo màu trước khi quay. |
| Người quá nhỏ trong khung | Đặt camera gần hơn: người nên cao ít nhất khoảng 1/3 khung hình. |
| Dev và test dùng cùng 4 người | Ghi rõ trong báo cáo rằng kết quả Re-ID sau khi chỉnh trọng số trên dev có thể lạc quan (XLA 3.1). |
