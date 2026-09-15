# Ghi chú triển khai (các phần chưa code)

Notebook hiện tại:

1. Giới thiệu
2. Phân tích dữ liệu MMOTU (2.1–2.12) — **đã làm**. Các chỉ số 2.5–2.12 tính trực tiếp từ ảnh/mask.
3. Tiền xử lý dữ liệu
   - **3.1 tách tập** — đã code (sklearn, 3 tỷ lệ → `mmotu_splits.csv`)
   - **3.2 làm sạch** — đã code (chỉ loại lỗi cứng; giữ outlier chất lượng)
   - **3.3 chuẩn hóa 384×384** — đã code (pad + CLAHE + min-max; mask nhị phân)
   - **3.4 class weight + augment nhẹ** — đã code (train OTU_2d, `split_8_1_1`)
4. Xây dựng mô hình — markdown, **code sau**
5. Huấn luyện và Đánh giá — markdown, **code sau**
6. Kết luận — markdown, **viết sau khi có kết quả train**

---

## 4. Mô hình (sẽ code)

- Dataset dùng `df_clean`, `preprocess_image` / `augment_train` (chỉ train), `class_weight`
- ImageNet mean/std sau khi ảnh đã về `[0, 1]`
- Backbone transfer learning — điền khi implement
