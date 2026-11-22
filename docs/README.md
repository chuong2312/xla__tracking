# Mục lục tài liệu

Gặp từ khó thì tra [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md).

## Đọc trước

| Tài liệu | Nội dung |
|---|---|
| [de_xuat_phuong_an.md](de_xuat_phuong_an.md) | Phương án: hệ thống làm gì, gồm những bước nào, rủi ro |
| [12_he_thong.md](12_he_thong.md) | Toàn bộ hệ thống chạy thế nào, kết quả trên video của nhóm |
| [15_du_lieu_that_va_dap_an.md](15_du_lieu_that_va_dap_an.md) | Dữ liệu thật của nhóm, cách kiểm tra khi chưa có đáp án, cách khoanh đáp án; chỗ đặt video: [data/README.md](../data/README.md) |
| [14_hieu_chinh_tham_so.md](14_hieu_chinh_tham_so.md) | Những chỗ chỉnh tham số, cách tự chấm điểm |
| [hoi_dap_bao_ve.md](hoi_dap_bao_ve.md) | Câu hỏi thầy có thể hỏi khi bảo vệ, kèm câu trả lời |

## Từng bước của hệ thống

| Bước | Tài liệu | Code |
|---|---|---|
| C. Tìm người | [10_tim_nguoi.md](10_tim_nguoi.md) | `src/detect/` |
| D. Bám người, giữ tên | [11_tracker_moi.md](11_tracker_moi.md), [04_kalman.md](04_kalman.md), [03_ghep_cap_hungarian.md](03_ghep_cap_hungarian.md) | `src/tracking/`, `src/common/` |
| Nối track bị đứt | [13_noi_tracklet.md](13_noi_tracklet.md) | `src/tracking/tracklet_linking.py` |
| F. Đặc trưng ngoại hình | [08_dac_trung_ngoai_hinh.md](08_dac_trung_ngoai_hinh.md) | `src/reid/` |
| G. Nhận dạng lại giữa 2 camera | [09_nhan_dang_lai.md](09_nhan_dang_lai.md), [02_homography_san.md](02_homography_san.md) | `src/reid/`, `src/common/homography.py` |
| H. Chấm điểm | [06_cham_diem_mota_idf1.md](06_cham_diem_mota_idf1.md) | `src/evaluation/` |
| Dữ liệu thật và đáp án | [15_du_lieu_that_va_dap_an.md](15_du_lieu_that_va_dap_an.md) | `tools/tao_canh.py`, `tools/annotate.py`, `tools/verify_gt.py` |

Số 05 và 07 (đếm VÀO/RA qua cửa) đã bỏ theo QĐ-21, nội dung cũ còn trong commit `827121c`. Số 01 (video máy vẽ) đã bỏ theo QĐ-31: dự án chỉ dùng video thật.

## Quản lý dự án

| Tài liệu | Nội dung |
|---|---|
| [quyet_dinh_thiet_ke.md](quyet_dinh_thiet_ke.md) | Nhật ký quyết định thiết kế QĐ-01 … QĐ-32: quyết định gì, vì sao, kết quả |
| [cau_hoi_thay.md](cau_hoi_thay.md) | Câu hỏi đã hỏi thầy và câu trả lời |
| [ke_hoach_quay_video.md](ke_hoach_quay_video.md) | Kế hoạch quay video thật |
| [tim_video_thuc_te.md](tim_video_thuc_te.md) | Yêu cầu và cách tìm video thật trên mạng |
| [tu_dien_thuat_ngu.md](tu_dien_thuat_ngu.md) | Từ điển thuật ngữ bằng tiếng Việt dễ hiểu |
