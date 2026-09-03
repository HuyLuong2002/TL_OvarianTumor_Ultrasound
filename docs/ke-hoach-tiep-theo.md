# Ghi chú triển khai (các phần chưa code)

Notebook hiện tại:

1. Giới thiệu
2. Phân tích dữ liệu MMOTU (2.1–2.12) — **đã làm**. Các chỉ số 2.5–2.12 tính trực tiếp từ ảnh/mask.
3. Tiền xử lý dữ liệu — markdown, **code sau**
4. Xây dựng mô hình — markdown, **code sau**
5. Huấn luyện và Đánh giá — markdown, **code sau**
6. Kết luận — markdown, **viết sau khi có kết quả train**

---

## 3. Tiền xử lý (sẽ code)

- Resize/pad kích thước cố định
- Chuẩn hóa intensity; cân nhắc CLAHE/HE
- Cân bằng lớp (3, 4, 6, 7 ít mẫu) và augment siêu âm
- Giữ split `train` / `val` theo `*_cls.txt`
