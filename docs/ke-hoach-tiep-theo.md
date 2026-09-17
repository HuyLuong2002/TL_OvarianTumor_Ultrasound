# Ghi chú triển khai

**Chạy notebook (local):** kernel `.venv` (Python 3.11 + PyTorch CUDA 11.7). Dữ liệu `dataset/MMOTU/OTU_2d/`, cache/checkpoint trong `cache/` và `runs/` cạnh notebook. GPU laptop (Quadro M1000M 2 GB): batch tự hạ + gradient accumulation; không dùng Colab/Drive.

Notebook hiện tại:

1. Giới thiệu
2. Phân tích dữ liệu MMOTU (2.1–2.12) — **đã làm**
3. Tiền xử lý dữ liệu
   - **3.1 tách tập** — đã code
   - **3.2 làm sạch** — đã code
   - **3.3 chuẩn hóa 384×384** — đã code
   - **3.4 class weight + augment nhẹ** — đã code (minh họa skimage)
   - **3.5 cache `.npy`** — đã code
4. Xây dựng mô hình — markdown + `build_resnet50`, freeze 2 giai đoạn — **đã code**
5. Huấn luyện và Đánh giá — Dataset/DataLoader, train 3 split, report/CM/curves — **đã code**
6. Kết luận — khung; **viết số liệu sau khi train xong** (`runs/results_splits.csv`)
