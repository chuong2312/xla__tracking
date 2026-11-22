# 01. Video tổng hợp v2: hành lang nhiều người đi lại

> Code: `tools/synthetic_scene.py` (các khối dựng cảnh), `tools/synthetic_scenarios.py` (bố trí và kịch bản),
> `tools/synthetic_render.py` (vẽ khung hình), `tools/make_synthetic_v2.py` (chạy chính).
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Video tổng hợp là video giả do máy tự vẽ: các hình chữ nhật có hoa văn đóng vai người đi lại trong hành lang. Vì máy tự vẽ nên **máy biết chính xác đáp án**: ai ở đâu, bị che bao nhiêu. Nhờ vậy không cần gán nhãn bằng tay, và có thể thử từng phần của chương trình trước khi có video thật.

## Bố trí (QĐ-22)

- **Hành lang** rộng 3 m, dài 10 m.
- **Hai camera đặt ở cùng một đầu hành lang:** `cam_a` ở góc trái, `cam_b` ở góc phải, cả hai nhìn dọc vào trong. Hai camera vì vậy cùng thấy gần hết hành lang, từ hai góc lệch nhau.
- Người ở cuối hành lang cao khoảng 40 pixel, ở giữa khoảng 70 pixel, sát camera khoảng 160 pixel.

## Cách làm

1. **Mọi thứ được đặt trên mặt sàn**, tính bằng mét. Mỗi người có các mốc "giây thứ mấy, đứng ở đâu"; giữa hai mốc thì đi thẳng đều.
2. **Mỗi camera có một homography sàn → ảnh** ([02](02_homography_san.md)), nên cùng một người có đáp án đúng ở cả hai camera.
3. **Kích thước người theo khoảng cách:** đo xem ở chỗ đó 1 m trên sàn dài bao nhiêu pixel, rồi nhân với chiều cao người.
4. **Che khuất tự nhiên:** vẽ người ở xa trước, người ở gần sau.
5. **Bản đồ "ai ở pixel nào":** sau khi vẽ, phần nhìn thấy của một người là các pixel còn mang mã của họ. Box đáp án chỉ bao phần nhìn thấy (XLA 3.3).
6. **2 giây đầu là cảnh trống**, để trừ nền trung vị có ảnh nền ban đầu.

## Công thức

- Vị trí giữa hai mốc (t0, P0) và (t1, P1): `P(t) = P0 + (t − t0)/(t1 − t0) · (P1 − P0)`.
- Tỉ lệ pixel trên mét: `s = √|det J|`, với J là đạo hàm của phép chiếu sàn → ảnh.
- Mức che theo tỉ lệ nhìn thấy r: r ≥ 90% là 0; 50–90% là 1; dưới 50% là 2; không thấy gì thì không ghi dòng.

## 5 kịch bản (4 người, 12 giây)

| Kịch bản | Nội dung | Thử phần nào |
|---|---|---|
| `s1` | Đi thưa theo hai chiều, mỗi người một làn | Trường hợp dễ |
| `s2` | Đi ngược chiều, đổi làn, cắt nhau và che nhau | Che khuất, tráo tên |
| `s3` | Người 1 đi vào rồi đứng yên 5 giây; người 2 đi ngang trước mặt | Người đứng yên |
| `s4` | Người 1 và người 2 mặc giống hệt nhau, đi ngược chiều cắt nhau | Không được dựa vào màu |
| `s5` | Người 1 và người 2 đi sát nhau thành nhóm rồi tách ra | Nhóm đi cùng |

## Chạy thử

```bash
python -m tools.make_synthetic_v2 --out data/synthetic_v2 --explain outputs/explain/video_tong_hop
```

Lệnh này mất khoảng 1 phút. Mỗi kịch bản ra một thư mục `data/synthetic_v2/<kịch bản>/` gồm:
- video và đáp án (`gt_tracking.csv`) của từng camera;
- `gt_reid.csv`: ai ở camera A là ai ở camera B;
- `scene.json`: homography và vị trí các điểm băng keo.

Ảnh minh họa nằm ở `outputs/explain/video_tong_hop/<kịch bản>.png`, với màu box: xanh là không che, vàng là che một phần, đỏ là che nặng.

## Khi nào không đủ

- Người là hình chữ nhật, không có tay chân, không có bóng, ánh sáng không đổi. Kết quả chỉ cho biết **code chạy đúng ý**, không phải chất lượng trên video thật.
- Mọi người đi thẳng đều. Tham số chọn trên video tổng hợp phải hiệu chỉnh lại trên dev thật.

## Thầy có thể hỏi

- *"Sao không dùng luôn video thật?"* Video tổng hợp có đáp án chính xác tuyệt đối và tạo lại được y hệt bằng cùng một seed, nên dùng để kiểm tra code. Video thật mới dùng để đánh giá chất lượng.
- *"Box đáp án khi bị che là box gì?"* Là box bao phần nhìn thấy, đúng quy ước gán nhãn của nhóm (XLA 3.3).
