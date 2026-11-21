# Tìm video thật trên mạng để thử chương trình

> **Mục đích:** có thêm video thật với nhiều điều kiện hành lang khác nhau, ngoài video nhóm tự quay ([ke_hoach_quay_video.md](ke_hoach_quay_video.md)).
> **Nhớ:** video trên mạng chỉ để **thử thêm**. Video nhóm tự quay mới đủ điều kiện cho toàn bộ đề tài: 2 camera chồng vùng nhìn, đồng bộ, có cửa, biết rõ ai là ai.

## 1. Điều kiện bắt buộc: thiếu một điều là không dùng được

Chương trình tìm người bằng cách so mỗi ảnh với ảnh nền. Vì vậy:

| Điều kiện | Vì sao |
|---|---|
| **Camera đứng yên tuyệt đối**: không cầm tay, không lia, không zoom, không rung | Camera nhúc nhích thì cả ảnh khác nền, chương trình thấy "chuyển động" ở khắp nơi (đúng lỗi của video S1 cũ) |
| **Không cắt cảnh** trong đoạn dùng để thử | Mỗi lần chuyển cảnh, ảnh nền đổi hoàn toàn |
| Dài ít nhất **15–30 giây liên tục** | Cần thời gian để có ảnh nền, và để người đi qua hết khung |
| Thấy **toàn thân người**, kể cả chân | Đếm cửa và homography dùng điểm chân |
| Người cao ít nhất khoảng **1/4 chiều cao khung** | Người quá nhỏ thì không đủ chi tiết |
| Camera đặt **cao, nhìn chéo xuống**, kiểu camera an ninh | Ít bị che hơn nhìn ngang, và thấy được sàn |
| Độ phân giải từ **480p** trở lên | Ảnh quá mờ thì không bám được |
| **Không có hiệu ứng**: filter, chữ chạy, sticker, chuyển cảnh | Những thứ này thay đổi trên ảnh và bị tưởng là chuyển động |

**Chữ ngày giờ chạy ở góc** (hay gặp ở video camera an ninh) thì vẫn dùng được: khai báo vùng đó trong `scene.ignore_regions` để chương trình bỏ qua.

## 2. Nguồn nên và không nên dùng

| Nguồn | Đánh giá |
|---|---|
| **Bộ dữ liệu nghiên cứu công khai** (mục 4) | **Tốt nhất**: camera cố định, có sẵn đáp án do người gán, có bộ còn có nhiều camera đồng bộ |
| **Kho video miễn phí** (Pexels, Pixabay…), tìm "CCTV", "security camera" | **Tốt**: nhiều đoạn quay cố định, giấy phép rõ ràng cho phép dùng |
| **YouTube**: bản ghi camera an ninh để nguyên, không cắt ghép | Dùng được nếu tìm được đoạn dài liên tục |
| **TikTok** | Thường **không dùng được**: quay dọc cầm tay, có hiệu ứng, chữ, cắt cảnh nhanh |
| **Phim** | Thường **không dùng được**: máy quay di chuyển liên tục và cắt cảnh vài giây một lần. Chỉ dùng được cảnh hiếm hoi giả làm màn hình camera an ninh |

**Bản quyền và quyền riêng tư:** video trên mạng có người lạ. Chỉ dùng để thử trong nhóm, ghi nguồn trong báo cáo, và không đăng lại. Kho video miễn phí và bộ dữ liệu nghiên cứu thì có giấy phép rõ ràng hơn; vẫn phải đọc điều khoản của từng nguồn.

## 3. Các điều kiện hành lang nên có

Mỗi video ghi lại nó thuộc ô nào trong bảng dưới. Mục tiêu là mỗi điều kiện có ít nhất 1–2 video, để báo cáo nói được "chương trình chạy tốt ở đâu, hỏng ở đâu".

| Nhóm | Điều kiện | Thử phần nào, thường hỏng thế nào |
|---|---|---|
| **Ánh sáng** | Sáng đều (văn phòng, đèn trần) | Trường hợp chuẩn để so sánh |
| | Ngược sáng: cửa kính hoặc cửa sổ cuối hành lang | Người thành bóng đen, khó tách khỏi nền |
| | Đèn tự bật/tắt theo cảm biến, nắng thay đổi | Ảnh nền đổi đột ngột, báo chuyển động khắp nơi |
| | Ban đêm, camera hồng ngoại (ảnh đen trắng) | Mất màu, nên phần so màu áo vô dụng |
| | Đèn huỳnh quang nhấp nháy | Nhiễu ở cả ảnh |
| **Sàn và tường** | Sàn bóng phản chiếu người | Phản chiếu bị tưởng là người thứ hai |
| | Bóng đổ rõ trên sàn | Kiểm tra phần khử bóng |
| | Tường kính, cửa kính | Phản chiếu, nhìn xuyên qua kính thấy người ở phía sau |
| **Người** | 1 người đi qua | Trường hợp dễ |
| | 2–4 người đi ngược chiều, cắt nhau | Che khuất, tráo tên |
| | Nhóm đi sát nhau, khoác vai | Nhiều người dính thành một khung |
| | Người đứng yên lâu (đứng chờ, nói chuyện điện thoại) | Bị "hòa" vào ảnh nền |
| | Mặc giống nhau (đồng phục học sinh, áo blouse bệnh viện) | Không phân biệt được bằng màu áo |
| | Mang đồ: balo, xe đẩy, vali | Khung bị to ra, điểm chân bị lệch |
| | Đông người (giờ tan học, giờ cao điểm) | Thử giới hạn của chương trình |
| **Cửa** | Người đi vào/ra một cửa trong khung hình | Đếm VÀO/RA (thầy bắt buộc) |
| | Cửa tự động hoặc cửa đóng mở liên tục | Cánh cửa chuyển động bị tưởng là người |
| | 2 người qua cửa cùng lúc, ngược chiều | Trường hợp khó của bộ đếm |
| **Góc quay** | Nhìn dọc hành lang (người đi về phía camera) | Người to dần lên, dễ che nhau |
| | Nhìn ngang hành lang (người đi ngang qua) | Ít che nhau, dễ hơn |
| | Nhìn gần như thẳng từ trên xuống | Người trông rất khác bình thường, chân khó xác định |
| **Chất lượng** | Camera an ninh cũ: mờ, nén nặng, 10–15 fps | Ít điểm để bám, chuyển động bị giật |

## 4. Bộ dữ liệu nghiên cứu công khai nên tìm

Tìm theo tên trên Google. Đây là các bộ dữ liệu quen thuộc trong giới nghiên cứu bám người; trước khi dùng, **đọc trang của bộ dữ liệu** để biết giấy phép, định dạng đáp án và đáp án được làm bằng cách nào. Đáp án phải do người gán, không được do mô hình học máy tạo ra (XLA 1).

| Bộ dữ liệu | Có gì | Hợp với phần nào |
|---|---|---|
| **CAVIAR** | Hành lang trung tâm thương mại, có đoạn quay **2 góc đồng bộ**, có đáp án khung + ID | Hành lang trong nhà, nhận dạng lại giữa 2 góc |
| **EPFL multi-camera pedestrian videos** | Nhiều camera **chồng vùng nhìn**, có cảnh trong nhà, có thông tin hình học mặt sàn | Gần giống bố trí của nhóm nhất: homography, Re-ID |
| **PETS 2009** | Nhiều camera quay người đi bộ ngoài trời | Bám người, nhiều góc nhìn |

Đáp án của các bộ này có định dạng riêng. Nếu dùng, mình sẽ viết công cụ chuyển sang định dạng `gt_tracking.csv` của nhóm.

## 5. Từ khóa tìm kiếm (copy dùng luôn)

**Tiếng Anh** (ra nhiều kết quả hơn):
- `CCTV hallway footage people walking`
- `security camera corridor footage fixed camera`
- `office corridor surveillance video`
- `school hallway CCTV footage`
- `hospital corridor security camera`
- `people entering door CCTV footage`
- `CCTV night infrared corridor`
- `crowded corridor surveillance footage`

Thêm `stock footage` hoặc `free` để ra kho video miễn phí.

**Tiếng Việt:**
- `camera an ninh hành lang`
- `camera giám sát hành lang trường học`
- `clip camera an ninh người đi qua cửa`
- `camera an ninh ban đêm hành lang`

**Câu hỏi mẫu nếu nhờ trợ lý AI tìm giúp:**
> "Tìm cho tôi các video quay bằng camera an ninh cố định ở hành lang trong nhà: camera không di chuyển, không cắt cảnh, dài ít nhất 20 giây, thấy toàn thân người đi lại, có người đi qua cửa. Ưu tiên nguồn có giấy phép dùng miễn phí hoặc bộ dữ liệu nghiên cứu có sẵn đáp án khung người."

## 6. Sau khi tải về

1. Cắt lấy đoạn **liên tục, không có chuyển cảnh**, dài 20–30 giây. Nên có vài giây đầu ít người, để có ảnh nền.
2. Đổi sang frame rate cố định: `ffmpeg -i goc.mp4 -vf fps=30 -an video.mp4`.
3. Đặt vào `data/real/internet/<tên_ngắn>/cam_a/video.mp4`, kèm một file `nguon.txt` ghi link, giấy phép và các điều kiện ở mục 3.
4. Video không có đáp án thì gán nhãn bằng `tools/annotate.py`. Giữ đoạn ngắn vì gán nhãn tốn công (xem [ke_hoach_quay_video.md](ke_hoach_quay_video.md), mục 8).
5. Video có chữ ngày giờ ở góc: khai báo vùng đó trong `scene.ignore_regions`.
