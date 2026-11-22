# Từ điển thuật ngữ

> Viết cho cả nhóm, nhất là người chưa học Xử lý ảnh.
> Mỗi dòng gồm hai phần: **tên hay gặp** (thường là tiếng Anh), sau đó là **nghĩa dễ hiểu**, có khi kèm ví dụ.
> Các từ được xếp theo thứ tự các bước của chương trình (xem mục 1 của [de_xuat_phuong_an.md](de_xuat_phuong_an.md)). Gặp từ mới chưa có ở đây thì thêm vào.

## Bức tranh chung trong 4 câu

1. Chương trình xem video và **tìm chỗ có người** (bước C).
2. Chương trình **bám theo từng người** qua các tấm ảnh, giữ cho mỗi người một cái tên riêng (bước D).
3. Chương trình **nối lại** những đoạn bám bị đứt của cùng một người.
4. Chương trình **nhận ra cùng một người** ở camera 1 và camera 2 (bước F, G). Sau đó nhóm **chấm điểm** chương trình bằng đáp án tự làm bằng tay (bước H).

## 1. Ảnh và video

| Từ | Nghĩa dễ hiểu |
|---|---|
| **pixel** (điểm ảnh) | Ô màu nhỏ nhất của ảnh. Ảnh 640×360 có 640 ô theo chiều ngang và 360 ô theo chiều dọc. |
| **frame** (khung hình) | Một tấm ảnh trong video. |
| **fps** | Số tấm ảnh trong mỗi giây video. Video 30 fps có 30 tấm mỗi giây. |
| **CFR / VFR** | CFR là số tấm mỗi giây cố định, VFR là thay đổi. Điện thoại hay quay kiểu VFR, làm hai video lệch giờ nhau dần dần, nên phải đổi sang CFR trước khi xử lý. |
| **độ phân giải, resize** | Độ phân giải là kích thước ảnh tính bằng pixel. Resize là thu nhỏ hoặc phóng to ảnh. "Giữ tỉ lệ" nghĩa là không bóp méo: video 16:9 thu về 640×360, chứ không phải 640×480. |
| **ảnh xám** (grayscale) | Ảnh chỉ có độ sáng, không có màu. Nhiều bước chạy trên ảnh xám cho nhanh. |
| **kênh màu, BGR** | Mỗi pixel màu gồm 3 con số: lượng xanh dương, xanh lá và đỏ. OpenCV xếp theo đúng thứ tự xanh dương – xanh lá – đỏ, nên gọi là BGR. |
| **HSV** | Một cách khác để ghi màu: H là màu gì (đỏ, vàng, xanh…), S là màu đậm hay nhạt, V là sáng hay tối. Vì tách riêng "tối đi" với "đổi màu", HSV giúp nhận ra bóng người dễ hơn. |
| **ROI** (vùng quan tâm) | Phần ảnh mà chương trình được phép xét. Mọi thứ ngoài vùng này bị bỏ qua, ví dụ ô cửa kính có phản chiếu. |
| **mask** (mặt nạ) | Ảnh đen trắng: chỗ trắng là "được xét" hoặc "có vật", chỗ đen là "bỏ qua" hoặc "không có gì". |
| **crop** | Cắt riêng phần ảnh chứa một người. |
| **chân máy** (tripod), **camera cố định** | Camera đứng yên tuyệt đối. Đây là điều bắt buộc, vì cách tìm người dựa vào việc nền không đổi. |

## 2. Tìm người (bước C)

| Từ | Nghĩa dễ hiểu |
|---|---|
| **nền** (background) | Ảnh khung cảnh lúc không có ai. |
| **trừ nền** (background subtraction) | So ảnh hiện tại với ảnh nền. Chỗ nào khác nhiều là có vật đang chuyển động. |
| **foreground** (vùng chuyển động) | Những pixel khác với nền. **Chưa chắc đó là người**: có thể là bóng, cánh cửa hay hình phản chiếu. |
| **ngưỡng** (threshold) | Mức dùng để quyết định "có" hay "không". Ví dụ: pixel khác nền hơn 30 đơn vị thì coi là foreground. |
| **trung vị** (median) | Số đứng giữa khi xếp các số từ nhỏ đến lớn. Năm lần đo 10, 11, 200, 12, 10 có trung vị là 11, không bị số 200 bất thường kéo lệch. Dùng để làm ảnh nền: một người đi ngang qua vài tấm ảnh không làm hỏng nền. |
| **median_running** | Tên cách trừ nền của nhóm. Ảnh nền ban đầu là trung vị của vài giây cảnh trống, sau đó được cập nhật dần. |
| **cập nhật có chọn lọc** | Chỉ cập nhật ảnh nền ở những chỗ **không có người**. Nhờ vậy người đứng yên lâu không bị "hòa" vào nền rồi biến mất. |
| **MOG2** | Cách trừ nền có sẵn trong OpenCV. Mỗi pixel nhớ vài màu hay gặp ở chỗ đó; ví dụ pixel ở chỗ lá cây đung đưa nhớ cả màu lá lẫn màu trời. Gặp màu lạ thì pixel đó là foreground. Thầy cho dùng. |
| **learningRate** (tốc độ cập nhật nền) | Nền "quên" ảnh cũ nhanh hay chậm. Nhanh quá thì người đứng yên bị nuốt vào nền. Chậm quá thì nền không theo kịp khi ánh sáng thay đổi. |
| **bóng, khử bóng** | Bóng người trên sàn làm sàn tối đi, nhưng màu gần như giữ nguyên. Dựa vào điều đó (trong HSV), chương trình nhận ra bóng và bỏ đi. |
| **morphology** (phép làm sạch hình dạng) | Các phép sửa lại mask: **erode** gọt viền cho co lại; **dilate** làm phình ra; **opening** xóa các chấm nhiễu lẻ tẻ; **closing** lấp lỗ và nối chỗ bị đứt. |
| **kernel** | "Cây cọ" dùng cho các phép làm sạch ở trên, ví dụ hình chữ nhật 3×9. Kernel dựng đứng 3×9 nối được thân với chân mà ít làm dính hai người đứng cạnh nhau. |
| **connected components** (vùng liên thông) | Gom các pixel trắng dính liền nhau thành từng cụm. Mỗi cụm cho ra một khung. |
| **blob** | Một cụm foreground. **Blob gộp**: hai người đứng sát nhau nên thành một cụm. **Blob tách**: một người bị vỡ thành hai cụm, ví dụ thân và chân. |
| **box** (khung) | Khung chữ nhật bao quanh một người. Được ghi bằng 4 số `(x, y, w, h)`: `x, y` là vị trí góc trên bên trái, `w` là chiều rộng, `h` là chiều cao. |
| **detection** (một lần phát hiện) | Một khung tìm được trong một tấm ảnh. Lúc này chưa biết đó là ai. |
| **detector** | Phần chương trình làm bước tìm người (bước C). |
| **mảnh người** (fragment) | Một phần của người bị tách riêng, ví dụ chỉ có đầu hoặc chỉ có chân, vì áo trùng màu nền nên phần thân "biến mất" trong mask. |
| **track trùng** (duplicate track) | Một người mà có hai track, tức hai tên, cùng lúc. Thường do một mảnh của người đó (đầu hoặc chân) bị coi là người mới. |
| **mô hình chiều cao** (height model) | Đường thẳng cho biết người đứng ở vị trí chân y thì cao khoảng bao nhiêu pixel: `cao = a·y + b`. Camera nhìn từ trên xuống nên người càng gần thì chân càng thấp trong ảnh và người càng cao. |
| **phân vị 90%** (percentile) | Xếp các số từ nhỏ đến lớn, lấy số đứng ở vị trí 90%. Ví dụ 100 chiều cao thì lấy số thứ 90. Mảnh người thấp nằm ở nửa dưới, nên phân vị 90% gần với chiều cao người nguyên vẹn. |
| **Theil–Sen** | Cách kẻ một đường thẳng qua một đám điểm mà không bị một vài điểm lạ kéo lệch: tính độ dốc của mọi cặp điểm rồi lấy **trung vị**. Đây là toán thống kê, không phải học máy. |
| **HOG + SVM** | Một cách tìm người bằng mô hình học máy đã huấn luyện sẵn. **Bị cấm** trong đồ án, sẽ xóa khỏi code. |

## 3. Bám người (bước D)

| Từ | Nghĩa dễ hiểu |
|---|---|
| **tracking** (bám theo) | Theo dõi từng người qua nhiều tấm ảnh và giữ nguyên tên của họ. |
| **track** | "Hồ sơ" của một người trong một camera: vị trí của người đó qua từng tấm ảnh, kèm một mã số. |
| **ID cục bộ / ID toàn cục** | ID cục bộ là mã số của một người trong một camera. ID toàn cục là mã số chung cho người đó ở cả hai camera. |
| **ID switch** (đổi tên nhầm) | Chương trình đang gọi An là số 1, giữa chừng lại gọi An là số 3, hoặc gán số của An cho Bình. Lỗi này càng ít càng tốt. |
| **optical flow** (luồng chuyển động) | Đo xem mỗi điểm trên ảnh dịch đi bao nhiêu giữa hai tấm ảnh liền nhau. |
| **Lucas-Kanade (LK)** | Cách đo luồng chuyển động **chỉ tại vài điểm đặc biệt**, nên chạy nhanh. Đồ án dùng LK để biết mỗi người đã đi được bao xa. |
| **Farneback** | Cách đo luồng chuyển động **cho mọi pixel**, chậm hơn LK nhiều. |
| **điểm góc** (corner, `goodFeaturesToTrack`) | Điểm có góc cạnh rõ, như góc logo trên áo. Bám theo điểm góc dễ hơn bám theo vùng trơn. |
| **vấn đề khẩu độ** (aperture problem) | Nhìn qua một lỗ nhỏ vào bức tường trơn đang trượt thì không biết tường trượt theo hướng nào. Vùng áo trơn cũng vậy, nên LK chỉ đo ở các điểm góc. |
| **kiểm tra đi rồi về** (forward-backward check) | Đo đường đi của một điểm từ tấm 1 sang tấm 2, rồi đo ngược từ tấm 2 về tấm 1. Nếu không quay về đúng chỗ cũ thì điểm đó không đáng tin và bị bỏ. |
| **trung vị độ dịch** | Lấy mức dịch chuyển "đứng giữa" của nhiều điểm, để vài điểm đo sai không kéo lệch kết quả. |
| **Kalman** (bộ lọc Kalman) | Đoán vị trí tiếp theo dựa vào vận tốc, rồi "trộn" dự đoán với vị trí đo được thật. Ví dụ: một chiếc xe đang chạy 10 m mỗi giây bị cây che mất 1 giây, ta đoán xe đã đi thêm 10 m. Kalman giúp bám theo người trong lúc họ bị che. |
| **dự đoán / hiệu chỉnh** (predict / correct) | Hai bước của Kalman: đoán trước, rồi sửa lại khi có số đo thật. |
| **Hungarian** (thuật toán phân công) | Ghép các khung vừa tìm được với những người đang theo dõi, sao cho **tổng độ lệch nhỏ nhất**. Giống như chia 4 việc cho 4 thợ sao cho tổng chi phí thấp nhất. |
| **greedy** (ghép tham lam) | Cặp nào hợp nhất thì ghép trước. Nhanh, nhưng có thể làm các cặp sau bị ghép sai. |
| **cost** (chi phí) | Con số đo mức "không hợp" giữa hai thứ. Càng nhỏ càng hợp. |
| **IoU** (độ chồng khớp) | Diện tích phần hai khung chồng lên nhau, chia cho diện tích của cả hai khung gộp lại. 1 là trùng khít, 0 là không chạm nhau. |
| **gating** (chặn cặp vô lý) | Cặp nào lệch quá xa thì cấm ghép, kể cả khi thuật toán phân công muốn ghép. |
| **trạng thái track** | Mỗi track luôn ở một trong năm trạng thái: **tentative** (mới thấy, chưa chắc là người thật); **confirmed** (đã chắc); **merged** (đang dính chung một cụm với người khác); **static** (đang đứng yên); **lost** (mất dấu, track kết thúc). |
| **occlusion** (che khuất) | Một người bị người khác hoặc đồ vật che mất. |
| **template / NCC** (so ảnh mẫu) | Lưu một ảnh nhỏ của người ngay trước khi bị che. Khi hai người tách ra, tìm chỗ giống ảnh đó nhất để biết ai là ai. |
| **histogram màu** | Bảng đếm xem mỗi màu xuất hiện bao nhiêu lần, tức là "phân bố màu quần áo". |
| **Bhattacharyya, tương quan histogram** | Hai cách đo xem hai histogram giống nhau đến mức nào. |

## 5. Đặc trưng ngoại hình (bước F)

| Từ | Nghĩa dễ hiểu |
|---|---|
| **đặc trưng** (feature) | Những con số mô tả ngoại hình, dùng để so hai người với nhau. |
| **keypoint** (điểm đặc trưng) | Điểm nổi bật trên ảnh, như góc logo, nút áo hay nếp gấp. |
| **descriptor** (mô tả điểm) | Dãy số mô tả vùng ảnh quanh một keypoint. Hai keypoint giống nhau thì hai dãy số gần nhau. |
| **SIFT** | Một cách tìm keypoint và tính descriptor. Vẫn nhận ra cùng một điểm khi ảnh bị phóng to, thu nhỏ hay xoay. |
| **ORB** | Giống SIFT nhưng nhanh hơn và kém chính xác hơn. FAST và BRIEF là hai bộ phận bên trong ORB. |
| **ASIFT** | SIFT có thêm bước "giả lập nhìn từ nhiều góc nghiêng", nên vẫn khớp được khi hai camera nhìn lệch góc nhau. Chậm hơn SIFT. |
| **sift_auto** | SIFT tự hạ ngưỡng khi ảnh người quá nhỏ và tìm được quá ít điểm. |
| **contrastThreshold** | Ngưỡng độ tương phản để SIFT giữ lại một điểm. Hạ ngưỡng thì được nhiều điểm hơn, nhưng các điểm thêm vào kém nổi bật hơn. |
| **gallery** (bộ ảnh mẫu) | Vài ảnh đẹp nhất của mỗi người (không bị che, không chạm mép ảnh), dùng để so sánh giữa hai camera. |
| **cân bằng trắng, gray-world** | Hai điện thoại chỉnh màu khác nhau, nên cùng một chiếc áo có thể hơi khác màu ở hai camera. Gray-world chỉnh màu sao cho màu trung bình của cả cảnh là màu xám, nhờ vậy hai camera cho màu gần giống nhau. |

## 6. Nhận ra cùng một người ở hai camera (bước G)

| Từ | Nghĩa dễ hiểu |
|---|---|
| **Re-ID** (nhận dạng lại) | Biết được người ở camera 1 chính là người nào ở camera 2. |
| **matching** (khớp điểm) | Ghép mỗi keypoint ở ảnh A với keypoint giống nó nhất ở ảnh B. |
| **knnMatch, ratio test** | Với mỗi điểm, tìm 2 điểm giống nhất ở ảnh kia. Chỉ tin kết quả khi điểm giống nhất giống **hơn hẳn** điểm thứ hai. Đây là tìm kiếm, không phải học máy. |
| **ràng buộc dọc** | Điểm ở đầu phải khớp với điểm ở đầu, điểm ở chân phải khớp với điểm ở chân. Nhờ vậy loại được các cặp khớp vô lý. |
| **RANSAC** | Cách tìm quy luật chung giữa nhiều cặp khớp và bỏ các cặp sai. Cách làm: lấy ngẫu nhiên vài cặp để đoán ra một quy luật, lặp lại nhiều lần, rồi giữ quy luật được nhiều cặp đồng ý nhất. |
| **homography** (phép chiếu mặt sàn) | Một bảng 3×3 dùng để chuyển một điểm trên sàn ở ảnh camera A sang vị trí tương ứng ở ảnh camera B, giống "bản đồ quy đổi" giữa hai camera. Được tính từ ít nhất 4 điểm băng keo dán trên sàn. Chỉ là **thông tin bổ trợ** và có thể sai: khi sàn không phẳng, chân bị che, hai video lệch giờ, hoặc hai người đứng sát nhau. |
| **chồng vùng nhìn** | Hai camera cùng nhìn một khu vực, từ hai góc khác nhau. |
| **đồng bộ** | Làm cho hai video khớp giờ với nhau, ví dụ vỗ tay một cái lúc bắt đầu để làm mốc. |
| **ma trận tương đồng** | Bảng điểm: mỗi ô cho biết người thứ i ở camera A giống người thứ j ở camera B đến mức nào. |
| **trọng số** (w1, w2, w3) | Mức quan trọng của từng loại điểm (keypoint, màu, vị trí trên sàn) khi cộng lại thành điểm tổng. |
| **tracklet, nối tracklet** | Tracklet là một đoạn track bị đứt. Nối tracklet là nối các đoạn của cùng một người lại với nhau, trước khi so sánh giữa hai camera. |
| **global_id = −1** | Nghĩa là "không khớp với ai": chương trình không đủ chắc để ghép người này với ai ở camera kia. |

## 7. Chấm điểm (bước H)

| Từ | Nghĩa dễ hiểu |
|---|---|
| **ground truth, gán nhãn** (đáp án) | Nhóm tự vẽ khung và ghi tên từng người bằng tay, dùng làm đáp án để chấm điểm chương trình. Chương trình không bao giờ nhìn vào đáp án khi chạy. |
| **keyframe, nội suy** | Keyframe: cứ vài tấm ảnh thì nhóm vẽ khung bằng tay cho một tấm. Nội suy: với các tấm ở giữa, máy tự đoán vị trí khung dựa vào hai keyframe hai bên. |
| **occluded = 0/1/2** | Ghi chú khi gán nhãn: 0 là không bị che, 1 là bị che một phần, 2 là bị che nặng. |
| **dev / test** | Dev là video để chỉnh thông số. Test là video để thi thật, không được dùng để chỉnh, để điểm số không bị "học tủ". |
| **hiệu chỉnh tham số, quét lưới** (grid search) | Thử vài giá trị cho mỗi thông số, rồi chọn giá trị cho điểm cao nhất trên video dev. **Không gọi là "huấn luyện"**. |
| **TP / FP / FN** | TP: tìm đúng. FP: báo nhầm, tức khung đó không phải người. FN: bỏ sót, tức có người mà chương trình không tìm ra. |
| **precision** (độ chính xác) | Trong các khung chương trình báo, bao nhiêu phần trăm là người thật. |
| **recall** (độ phủ) | Trong số người thật, chương trình tìm ra được bao nhiêu phần trăm. |
| **MOTA** | Điểm tổng của bước bám người, tính cả bỏ sót, báo nhầm và đổi tên nhầm. Chủ yếu phản ánh việc "tìm có đủ người không". |
| **MOTP** | Trung bình mức khớp giữa các khung tìm đúng và khung trong đáp án. |
| **IDF1** | Điểm cho việc **giữ đúng tên** của mỗi người suốt cả video. |
| **rank-1** | Với mỗi người ở camera A, xem người giống nhất ở camera B có đúng là họ không. |
| **x/y cặp đúng** | Ví dụ 3/4 nghĩa là ghép đúng 3 trong 4 người. Luôn ghi rõ mẫu số. |
| **oracle** (chạy với đáp án) | Cho một bước dùng khung trong đáp án thay vì khung tự tìm. Nhờ vậy đo được riêng năng lực của bước sau, không bị lỗi của bước trước ảnh hưởng. |
| **end-to-end** (từ đầu đến cuối) | Chạy toàn bộ chương trình thật. Lỗi của bước trước cộng dồn sang bước sau. |
| **ablation** (thí nghiệm so sánh) | Tắt hoặc thay một bộ phận để xem bộ phận đó giúp được bao nhiêu. Ví dụ: so MOG2 với trung vị, so SIFT với ASIFT. |
| **failure gallery** (bộ ca sai) | Ảnh các trường hợp chương trình làm sai, kèm giải thích vì sao sai. |
| **benchmark** | Đo tốc độ (mỗi tấm ảnh mất mấy mili giây) và lượng bộ nhớ dùng. |
| **RSS** | Lượng RAM mà chương trình đang thật sự chiếm. Dùng RSS thay cho `tracemalloc`, vì `tracemalloc` không thấy phần RAM do OpenCV dùng. |

## 8. Học máy (để trả lời thầy)

| Từ | Nghĩa dễ hiểu |
|---|---|
| **ML** (học máy) | Máy tự rút ra quy luật từ nhiều ví dụ có sẵn đáp án. Đồ án **không dùng**. |
| **DL** (học sâu) | Học máy dùng mạng nơ-ron nhiều lớp. Đồ án **không dùng**. |
| **YOLO** | Một mô hình học sâu rất phổ biến để tìm vật trong ảnh. **Bị cấm** trong đồ án. |
| **huấn luyện** (train) | Cho mô hình học từ dữ liệu có đáp án. Đồ án không có bước này, nên không dùng từ "huấn luyện". |
| **KNN** (thuật toán phân loại) | Phân loại một thứ dựa vào các ví dụ gần nó nhất; đây là học máy. **Khác** với `knnMatch`, vốn chỉ là tìm điểm gần nhất. |
| **k-means** | Cách để máy tự chia dữ liệu thành nhóm; đây là học không giám sát. Đồ án **không dùng**. |

## 9. Công cụ và file

| Từ | Nghĩa dễ hiểu |
|---|---|
| **OpenCV, NumPy, SciPy** | Ba thư viện Python: OpenCV để xử lý ảnh, NumPy để tính toán trên mảng số, SciPy cho toán khoa học (có sẵn hàm Hungarian). |
| **pipeline** (dây chuyền) | Chuỗi các bước nối tiếp nhau: tìm người → bám người → nối track bị đứt → nhận dạng lại → chấm điểm. |
| **module** | Một phần của chương trình chuyên làm một việc, thường là một file trong thư mục `src/`. |
| **config.yaml** | File chứa tất cả các "nút vặn" (ngưỡng, kích thước…). Muốn đổi thông số thì sửa số trong file này, không cần sửa code. |
| **CSV** | File bảng dạng chữ, mỗi dòng là một bản ghi. Ví dụ: `detections.csv` (các khung tìm được), `tracks.csv` (vị trí từng người), `gt_tracking.csv` (đáp án). |
| **`--explain`** | Lệnh sẽ có ở mỗi module: xuất ảnh minh họa từng bước, để nhóm dễ hiểu và đưa vào báo cáo. |

## 10. Từ thêm khi làm các module 01–06

| Từ | Nghĩa dễ hiểu |
|---|---|
| **video tổng hợp** | Video giả do máy tự vẽ: hình chữ nhật có hoa văn đóng vai người. Máy biết sẵn đáp án nên không cần gán nhãn tay. |
| **kịch bản** (`s1`–`s5`) | Mỗi kịch bản là một cách đi lại của 4 người, dùng để thử một tình huống: đi thưa, cắt nhau, đứng yên, mặc giống nhau, đi thành nhóm. |
| **mốc, quỹ đạo** | Quỹ đạo là đường đi của một người. Mốc là các điểm "giây thứ mấy, đứng ở đâu"; giữa hai mốc người đi thẳng đều. |
| **seed** | Con số khởi đầu cho phần ngẫu nhiên. Cùng seed thì máy vẽ ra y hệt nhau, nên kết quả kiểm tra lặp lại được. |
| **bản đồ ai ở pixel nào** (label map) | Bảng cùng kích thước ảnh, mỗi ô ghi mã người đang hiện ở pixel đó. Dùng để biết phần nhìn thấy của từng người khi bị che. |
| **pixel trên mét** | Ở một chỗ trên sàn, 1 m dài bao nhiêu pixel trong ảnh. Gần camera thì lớn, xa camera thì nhỏ. |
| **điểm băng keo** | Các điểm dán trên sàn, nhìn thấy ở cả hai camera, dùng để tính homography. |
| **sai số chiếu lại** | Khoảng cách (pixel) giữa điểm chiếu từ camera A sang và điểm thật ở camera B. Càng nhỏ thì homography càng chuẩn. |
| **chi phí cấm** (`FORBIDDEN_COST`) | Số rất lớn (một triệu) gán cho cặp không được ghép, rồi bị loại sau khi ghép. |
| **hệ số tin K** (Kalman gain) | Trong Kalman: tin phép đo bao nhiêu so với dự đoán, nằm giữa 0 (bỏ qua phép đo) và 1 (tin hoàn toàn). |
| **test hồi quy** | Test giữ lại mãi để chắc chắn lần sửa code sau không làm hỏng điều đã đúng, ví dụ test "hai người đi qua nhau vẫn giữ đúng tên" của tracker. |
| **CLEAR MOT** | Tên bộ quy tắc chuẩn để chấm MOTA và MOTP. |
| **IDTP / IDFP / IDFN** | Khi chấm IDF1: số box mang đúng tên / số box của chương trình mang sai tên / số box trong đáp án không được gọi đúng tên. |
| **frame_offset** | Số frame chênh lệch giữa hai video do bấm quay không cùng lúc; tìm bằng tiếng vỗ tay, rồi cộng vào để hai video cùng một mốc giờ. |
| **mức đủ** (`feature_saturation_matches`) | Số cặp keypoint khớp tốt mà từ đó trở lên coi là chắc chắn (điểm keypoint = 1). |
| **điểm gộp / điểm tổng** | Trung bình có trọng số của điểm keypoint, điểm màu và điểm vị trí sàn. |
| **"chỉ ngoại hình" / "ngoại hình + sàn"** | Hai cách gộp điểm, luôn báo cáo riêng: không dùng vị trí sàn, và có dùng vị trí sàn. |
| **người đứng trước / đứng sau** | Khi hai người dính nhau, người có chân thấp hơn trong ảnh là người gần camera hơn (đứng trước), nên vẫn được nhìn thấy đầy đủ. |
| **khung khả nghi** | Box mới to hoặc nhỏ bất thường so với người đang theo dõi và chồng lên người khác; tracker không lấy vị trí, kích thước từ box đó. |
| **bỏ phiếu đa số** | Khi chấm điểm: mỗi track được gắn với người trong đáp án mà nó khớp nhiều tấm ảnh nhất. |
| **track chính** | Nếu một người bị tách thành nhiều track, track khớp với người đó lâu nhất là track chính. |
| **end-to-end** (chạy trọn) | Chạy cả hệ thống thật: lỗi của bước trước cộng dồn sang bước sau. Ngược với oracle (dùng khung đáp án). |
| **bố trí hành lang hai góc** | Hai camera đặt ở cùng một đầu hành lang, một ở góc trái, một ở góc phải, cùng nhìn dọc vào trong (QĐ-22). |
| **nối tracklet** | Sau khi bám người xong, nối các đoạn track bị đứt của cùng một người: đoạn sau bắt đầu sau đoạn trước, ở gần chỗ đoán theo vận tốc, và cùng màu áo. |
| **ngưỡng có độ trễ** (hysteresis) | Dùng hai ngưỡng khác nhau cho "vào" và "ra", để trạng thái không chập chờn khi giá trị dao động quanh một ngưỡng. Ví dụ: vào cụm gộp cần 60%, ở lại chỉ cần 30%. |
| **tráo tên** (ID switch) | Hai người bị đổi tên cho nhau, hoặc một người bị gán tên của người khác. |
