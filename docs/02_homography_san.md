# 02. Homography mặt sàn

> Code: `src/homography.py` (tính toán), `tools/calibrate_floor.py` (click điểm và xuất ảnh kiểm tra).
> Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Bài toán

Hai camera cùng nhìn một khu vực từ hai góc khác nhau. Muốn biết "người đang đứng ở chỗ này trong camera A thì ở camera B họ đứng chỗ nào", ta cần một **bản đồ quy đổi vị trí trên sàn** giữa hai ảnh. Bản đồ đó là homography. Thông tin này dùng ở bước nhận dạng lại (G), và chỉ là **thông tin bổ trợ** (QĐ-02).

## Cách làm

1. Trước khi quay, dán ít nhất 4 điểm băng keo trên sàn, rải rộng và không thẳng hàng.
2. Chạy `tools/calibrate_floor.py`. Click các điểm băng keo ở ảnh camera A (nửa trái), rồi click **theo đúng thứ tự đó** ở ảnh camera B (nửa phải).
3. Chương trình tính ma trận H bằng `cv2.findHomography`:
   - đúng 4 cặp điểm: giải trực tiếp;
   - hơn 4 cặp: dùng RANSAC để tự loại cặp bị click nhầm.
4. Kết quả lưu ra file JSON, kèm **sai số chiếu lại** của từng điểm để biết click có chuẩn không.

## Công thức

H là ma trận 3×3. Với điểm (x, y) ở ảnh A:

```
[u, v, w] = H · [x, y, 1]
x_B = u / w,  y_B = v / w
```

H có 8 ẩn (ma trận 9 số, nhưng nhân H với cùng một số thì kết quả không đổi). Mỗi cặp điểm cho 2 phương trình, nên cần ít nhất 4 cặp.

**Sai số chiếu lại** của một điểm là khoảng cách (pixel) giữa điểm A sau khi chiếu sang B và điểm B mình đã click. Click chuẩn thì sai số khoảng 1–2 px.

**RANSAC:** lấy ngẫu nhiên 4 cặp, tính H, đếm xem bao nhiêu cặp khác "đồng ý" với H đó (lệch dưới ngưỡng). Lặp nhiều lần, giữ H được nhiều cặp đồng ý nhất. Cặp click nhầm sẽ không đồng ý với H đúng nên bị loại.

## Tham số (`config.yaml`)

| Tham số | Ý nghĩa |
|---|---|
| `homography.ransac_threshold_px` (3.0) | Cặp điểm chiếu lệch quá bấy nhiêu pixel thì bị coi là click nhầm |

## Chạy thử

```bash
# Click điểm (mở cửa sổ)
python tools/calibrate_floor.py --cam-a <video_a> --cam-b <video_b> --out <homography.json>
# Xuất ảnh kiểm tra: điểm đỏ (chiếu từ A) phải trùng điểm xanh (click ở B)
python tools/calibrate_floor.py --cam-a data/synthetic_v2/s2/cam_a/video.mp4 \
    --cam-b data/synthetic_v2/s2/cam_b/video.mp4 \
    --out data/synthetic_v2/s2/homography.json --explain outputs/explain/homography_san/s2.png
```

Trên video tổng hợp, H tính từ 6 điểm băng keo trùng với H thật, sai số khoảng 0,001 px. Test `tests/test_homography.py` còn cố ý làm click lệch một điểm 47 px, và RANSAC phát hiện ra đúng điểm đó.

## Video thật EPFL: dùng hiệu chỉnh có sẵn

EPFL đã đo sẵn cho mỗi camera một ma trận G "ảnh → bản đồ sàn nhìn từ trên xuống" (file `calibration-4p.txt`). Không cần click tay:

```
H_AB = G_B⁻¹ · G_A      (ảnh A → bản đồ sàn → ảnh B)
python -m tools.epfl_homography --calib data/real/epfl_4p_c0_c1/calibration-4p.txt --cam-a 0 --cam-b 1 \
    --scene data/real/epfl_4p_c0_c1 --explain outputs/explain/homography_epfl.png
```

Kiểm tra: ảnh minh họa vẽ cùng một lưới sàn lên hai camera, và lưới nằm phẳng trên sàn ở cả hai. Chiếu điểm chân người (lấy từ track) từ A sang B: trung vị cách người gần nhất ở B là 37 px, so với 100 px nếu chọn điểm ngẫu nhiên. Sai số còn lại chủ yếu do box người chưa chuẩn (chỉ có đầu, chân bị che), nên vị trí sàn vẫn chỉ là thông tin bổ trợ.

## Khi nào hỏng

- **Sàn không phẳng** (bậc thềm, dốc): homography chỉ đúng cho một mặt phẳng.
- **Chân bị che hoặc box lệch:** điểm chân đưa vào bị sai nên vị trí chiếu sang cũng sai.
- **Hai video lệch giờ:** so vị trí của người ở hai thời điểm khác nhau.
- **Hai người đứng sát nhau:** vị trí chiếu sang gần cả hai người.
- **4 điểm gần thẳng hàng** hoặc dồn vào một góc sàn: H tính ra kém chính xác ở các chỗ khác.

## Thầy có thể hỏi

- *"Vì sao cần ít nhất 4 điểm?"* H có 8 ẩn, mỗi cặp điểm cho 2 phương trình.
- *"Homography có luôn ghép đúng người không?"* Không. Đây là thông tin bổ trợ, có thể sai trong các trường hợp ở mục trên. Vì vậy nhóm báo cáo riêng kết quả chỉ dùng ngoại hình.
- *"RANSAC ở đây khác gì RANSAC khi khớp keypoint trên người?"* Ở đây các điểm nằm trên mặt sàn, là một mặt phẳng thật, nên giả định của homography đúng. Còn người không phải vật phẳng, cứng, nên RANSAC trên keypoint người chỉ dùng để so sánh (XLA 2.8).
