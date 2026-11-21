# 01. Video tổng hợp v2

> Code: `tools/synthetic_scene.py` (mô tả cảnh), `tools/synthetic_render.py` (vẽ khung hình), `tools/make_synthetic_v2.py` (chạy chính).
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Nhóm chưa có video thật, nhưng vẫn cần dữ liệu để thử từng phần của chương trình. Video tổng hợp là video giả do máy tự vẽ: các hình chữ nhật có hoa văn đóng vai người đi lại. Vì máy tự vẽ nên **máy biết chính xác đáp án**: ai ở đâu, bị che bao nhiêu, lúc nào đi qua cửa. Nhờ vậy không cần gán nhãn bằng tay.

## Cách làm

1. **Mọi thứ được đặt trên mặt sàn**, tính bằng mét: sàn rộng 6 m, sâu 4 m. Mỗi người có một danh sách mốc "giây thứ mấy, đứng ở đâu"; giữa hai mốc thì người đi thẳng đều.
2. **Mỗi camera có một homography "sàn → ảnh"** (xem [02_homography_san.md](02_homography_san.md)). Biết người đứng ở đâu trên sàn thì tính được điểm chân của họ trong ảnh của từng camera. Cùng một người nên có đáp án đúng ở cả hai camera.
3. **Kích thước người theo khoảng cách:** đứng gần camera thì to, đứng xa thì nhỏ. Chương trình đo xem ở chỗ đó 1 m trên sàn dài bao nhiêu pixel trong ảnh, rồi nhân với chiều cao người.
4. **Che khuất tự nhiên:** người ở xa được vẽ trước, người ở gần vẽ sau đè lên, giống ngoài đời.
5. **Bản đồ "ai ở pixel nào":** song song với ảnh màu, máy giữ một bảng cùng kích thước, mỗi ô ghi mã người đang hiện ở pixel đó. Vẽ xong, phần nhìn thấy của một người là các ô còn mang mã của họ. Box đáp án chỉ bao phần nhìn thấy (XLA 3.3).
6. **2 giây đầu là cảnh trống**, để cách trừ nền trung vị có ảnh nền ban đầu.

## Công thức

- Vị trí giữa hai mốc (t0, P0) và (t1, P1): `P(t) = P0 + (t − t0)/(t1 − t0) · (P1 − P0)`.
- Tỉ lệ pixel trên mét ở một chỗ trên sàn: `s = √|det J|`, với J là ma trận đạo hàm của phép chiếu sàn → ảnh, tính xấp xỉ bằng cách dịch 5 cm theo mỗi trục.
- Mức che theo tỉ lệ nhìn thấy `r = số pixel còn thấy / diện tích box nằm trong ảnh`:
  - r ≥ 90%: mức 0 (không che);
  - 50% ≤ r < 90%: mức 1 (che một phần);
  - r < 50%: mức 2 (che nặng);
  - không còn pixel nào: không ghi dòng.

## Hai kiểu bố trí camera

- **`san_chung`** (dùng cho `s1`–`s4` và `door`): sàn 6 m × 4 m, hai camera cùng nhìn cả khu vực từ hai góc.
- **`hanh_lang`** (dùng cho `hanh_lang_cua`): hành lang dài 8 m, có cửa ở tường bên trái. Camera A ở đầu hành lang nhìn dọc vào trong; camera B gắn đối diện cửa. Hai camera chỉ cùng thấy vùng quanh cửa.

Dữ liệu của các bố trí và kịch bản nằm ở `tools/synthetic_scenarios.py`; muốn thêm kịch bản thì sửa file này.

## Các kịch bản

| Kịch bản | Nội dung | Thử phần nào |
|---|---|---|
| `s1` | 4 người đi 4 làn riêng, lệch giờ nhau | Trường hợp dễ |
| `s2` | Đi chéo, cắt ngang nhau | Che khuất, hai người dính thành một cụm |
| `s3` | Người 1 đi vào giữa, đứng yên 5 giây rồi đi tiếp; người 3 đi ngang qua trước mặt | Người đứng yên |
| `s4` | Người 1 và người 2 mặc giống hệt nhau, đi cắt nhau | Không được dựa vào màu áo |
| `door` | 2 người vào, 2 người ra; người 3 và người 4 qua cửa gần như cùng lúc, ngược chiều | Đếm vào/ra |
| `hanh_lang_cua` | **Bố trí hành lang + cửa** (QĐ-17): người 1 rẽ vào cửa; người 4 vào sau 1,3 giây; người 3 từ phòng đi ra; người 2 đi ngang qua cửa mà không vào | Đếm vào/ra, ghép hai camera theo thời gian qua cửa ([07](07_ghep_su_kien_cua.md)) |

## Chạy thử

```bash
python -m tools.make_synthetic_v2 --out data/synthetic_v2 --explain outputs/explain/video_tong_hop
```

Lệnh này mất khoảng 1 phút. Mỗi kịch bản ra một thư mục `data/synthetic_v2/<kịch bản>/` gồm:
- video của hai camera;
- đáp án của từng camera (`gt_tracking.csv`);
- `gt_reid.csv`: ai ở camera A là ai ở camera B;
- `gt_events.csv`: các lượt qua cửa;
- `scene.json`: hình học của cảnh.

Ảnh minh họa nằm ở `outputs/explain/video_tong_hop/<kịch bản>.png`, với màu box: xanh là không che, vàng là che một phần, đỏ là che nặng.

**Lưu ý:** chạy `tools/verify_gt.py` trên `s2` sẽ có một cảnh báo "lỗ hổng frame" cho người 1. Cảnh báo này **đúng**: ở frame 165, người 1 bị che hoàn toàn nên frame đó không có box.

## Khi nào không đủ

- Người là hình chữ nhật, không có tay chân, không có bóng, ánh sáng không đổi. Kết quả trên video tổng hợp chỉ cho biết **code chạy đúng ý**, không cho biết chương trình tốt đến đâu trên video thật.
- Mọi người đi thẳng đều, không rẽ, không tăng giảm tốc. Các tham số chọn trên video tổng hợp (ví dụ của Kalman) phải hiệu chỉnh lại trên video dev thật.

## Thầy có thể hỏi

- *"Sao không dùng luôn video thật?"* Video tổng hợp có đáp án chính xác tuyệt đối và tạo lại được giống hệt bằng cùng một seed, nên dùng để kiểm tra code có đúng không. Video thật mới dùng để đánh giá chất lượng.
- *"Box đáp án khi bị che là box gì?"* Là box bao phần nhìn thấy, đúng quy ước gán nhãn của nhóm (XLA 3.3). Phần bị che không được đoán thêm.
