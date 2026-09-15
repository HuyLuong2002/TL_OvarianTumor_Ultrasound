# Thiết kế mục 4–5: Phân loại u buồng trứng bằng transfer learning (ResNet50)

Ngày: 2026-09-15
Trạng thái: đã được duyệt, chờ lập kế hoạch triển khai

## 1. Bối cảnh

Notebook `transfer-learning-ovarian-tumor-ultrasound.ipynb` đã hoàn thành mục 1–3:

- Mục 2: EDA đầy đủ trên **OTU_2d** (1469 ảnh, 8 lớp), xuất `mmotu_dataset_summary.csv` và `mmotu_feature_summary.csv`.
- Mục 3.1: chia lại train/val/test theo 3 tỷ lệ, xuất `mmotu_splits.csv` với các cột `split_8_1_1`, `split_7_1.5_1.5`, `split_6_2_2` (stratified theo `label`, `random_state=42`).
- Mục 3.2: làm sạch — không mẫu nào bị loại, `df_clean` giữ đủ 1469 mẫu.
- Mục 3.3: `preprocess_image` (pad vuông → resize 384×384 → CLAHE → min-max về `[0,1]`, ảnh xám 1 kênh) và `preprocess_mask`.
- Mục 3.4: `class_weight` balanced và `augment_train` (skimage) cho tập train của `split_8_1_1`.

Mục 4 (xây dựng mô hình), mục 5 (huấn luyện và đánh giá) và mục 6 (kết luận) hiện chỉ là markdown khung.

## 2. Phạm vi

Bài toán: **chỉ phân loại 8 lớp** trên OTU_2d bằng transfer learning. Không làm phân đoạn. Mask nhị phân chỉ dùng cho EDA (mục 2.6–2.7) và minh họa ở mục 3.3.

Ngoài phạm vi: phân đoạn, Dice/IoU, Grad-CAM hay bất kỳ hình thức giải thích mô hình nào, ensemble, tìm siêu tham số tự động, huấn luyện trên OTU_3d.

## 3. Các quyết định đã chốt

| Hạng mục | Quyết định |
| --- | --- |
| Bài toán | Phân loại 8 lớp, không phân đoạn |
| Framework | PyTorch + torchvision |
| Backbone | ResNet50, weight ImageNet-1k (`IMAGENET1K_V2`) |
| So sánh kiến trúc | Chỉ so sánh trên giấy + số tham số in bằng code cho ResNet50, EfficientNet-B0, DenseNet121; **chỉ train ResNet50** |
| Kích thước ảnh | 384×384 (giữ nguyên mục 3.3) |
| Thí nghiệm | 1 backbone × 3 tỷ lệ split đã có |
| Phần cứng | Colab hosted runtime, GPU T4 |
| Dữ liệu trên Colab | Cache `.npy` + CSV đặt trên Google Drive, mount mỗi session |
| Giải thích mô hình | Không làm |

## 4. Kiến trúc mô hình (mục 4)

### 4.1 Model chính

- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`. Dùng bộ weight V2 vì nó đạt 80.86% top-1 trên ImageNet so với 76.13% của V1, cùng kiến trúc nên không phải đổi gì trong code.
- Thay `fc` gốc (`Linear(2048, 1000)`) bằng `Sequential(Dropout(0.3), Linear(2048, 8))`.
- Đầu vào: ảnh xám 384×384 trong `[0,1]` lấy từ cache, nhân thành 3 kênh giống nhau, chuẩn hóa bằng mean/std ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`). Nhân 3 kênh là điều kiện để dùng được weight pretrained vốn học trên ảnh RGB.
- Số tham số: ResNet50 gốc có 25,557,032 tham số; sau khi thay head 8 lớp còn khoảng 23.5M. Con số chính xác sẽ được in bằng code, không chép tay.

### 4.2 Chiến lược fine-tune hai giai đoạn

**Giai đoạn 1 — warm-up head.** Đóng băng toàn bộ backbone, chỉ train `fc` (16,392 tham số). 5 epoch, lr 1e-3. Mục đích: head khởi tạo ngẫu nhiên sinh gradient lớn, nếu train chung ngay từ đầu sẽ làm nhiễu weight pretrained.

**Giai đoạn 2 — fine-tune phần sâu.** Mở `layer3`, `layer4` và `fc`; giữ đóng băng `conv1`, `bn1`, `layer1`, `layer2`. 25 epoch, lr 1e-4, `CosineAnnealingLR`. Số tham số học ở giai đoạn này khoảng 22.1M trên tổng 23.5M, phần đóng băng chỉ còn ~1.44M. Lý do giữ các tầng đầu: chúng học cạnh và texture cơ bản, dùng lại được cho ảnh siêu âm, và giảm số tham số phải học trên tập chỉ ~1000 ảnh train.

**Xử lý BatchNorm.** Các module bị đóng băng phải được đặt `.eval()` trong suốt quá trình train (không chỉ `requires_grad=False`), để BatchNorm không cập nhật `running_mean` / `running_var`. Nếu bỏ qua bước này, thống kê BN của phần backbone "đã đóng băng" vẫn trôi theo dữ liệu mới và làm mất tác dụng của việc đóng băng. Đây là lỗi thường gặp và phải được kiểm tra bằng một assert đếm số module đang ở chế độ train.

### 4.3 Bảng so sánh kiến trúc

Một cell khởi tạo cả 3 backbone (không train) và in bảng: tên, số tham số tổng, số tham số sau khi thay head 8 lớp, số chiều feature trước classifier (ResNet50 2048, EfficientNet-B0 1280, DenseNet121 1024).

Markdown kèm theo giải thích lý do chọn ResNet50: paper MMOTU gốc báo cáo baseline phân loại trên họ ResNet/VGG/DenseNet nên số của ta đối chiếu được trực tiếp; kiến trúc residual kinh điển, dễ trình bày trong báo cáo và có sẵn bộ weight `IMAGENET1K_V2` mạnh hơn hẳn V1. Đồng thời nêu rõ đánh đổi: ResNet50 có 23.5M tham số, nhiều hơn DenseNet121 (~7M) và EfficientNet-B0 (~4M), nên nguy cơ overfit trên ~1000 ảnh train cao hơn — đây chính là lý do phải đóng băng `conv1`/`layer1`/`layer2`, dùng dropout 0.3, augment nhẹ và early stopping theo macro-F1.

Kèm `torchinfo.summary` cho model chính (cài `torchinfo` bằng pip nếu thiếu).

## 5. Pipeline dữ liệu (mục 3.5 + 5.1)

### 5.1 Cache tiền xử lý — mục 3.5, chạy ở local

Chạy `preprocess_image` cho toàn bộ 1469 ảnh một lần, nhân 255 và ép về `uint8`, lưu:

- `cache/otu2d_384_u8.npy` — mảng `uint8` shape `(1469, 384, 384)`, khoảng 216 MB.
- `cache/otu2d_index.csv` — các cột `image_name`, `label`, `class_name`, `split_8_1_1`, `split_7_1.5_1.5`, `split_6_2_2`; **thứ tự dòng khớp đúng thứ tự phần tử trong mảng `.npy`**.

Hai file này được đẩy tay lên Drive vào `MyDrive/TL_OvarianTumor/cache/`, tức trên Colab sau khi mount sẽ là `/content/drive/MyDrive/TL_OvarianTumor/cache/`. Checkpoint và các CSV kết quả ghi vào `MyDrive/TL_OvarianTumor/runs/`. Thư mục `cache/` ở local phải được thêm vào `.gitignore` (216 MB, không đưa vào git); ngược lại `results_splits.csv` và `train_history_<split>.csv` vẫn theo git vì nhỏ và là số liệu cho báo cáo.

Lý do cache: `equalize_adapthist` (CLAHE) mất khoảng 0.1–0.2 giây mỗi ảnh. Nếu gọi trong `Dataset` thì mỗi epoch tốn thêm vài phút CPU, mà Colab chỉ có 2 vCPU nên GPU sẽ ngồi chờ dữ liệu. Cache một lần rồi nạp cả mảng vào RAM (Colab có ~12.7 GB) giúp mỗi epoch chỉ còn việc lấy slice từ RAM.

Ép về `uint8` làm mất tối đa 1/255 ≈ 0.4% giá trị cường độ, không ảnh hưởng tới kết quả và giúp file nhỏ hơn 4 lần so với `float32`.

Cell này phải in ra shape, dtype, dung lượng file và kiểm tra lại bằng cách so một mẫu random giữa cache và `preprocess_image` gọi trực tiếp.

### 5.2 Dataset và DataLoader — mục 5.1, chạy trên Colab

`Dataset` nhận mảng cache trong RAM cùng danh sách chỉ số của một split, trả về tensor `float32` shape `(3, 384, 384)` đã chuẩn hóa ImageNet và nhãn `int64`. Chia 255 và nhân 3 kênh thực hiện lúc lấy mẫu.

Augment chỉ áp cho train, giữ **đúng** bộ phép biến đổi đã chốt ở mục 3.4 nhưng viết lại bằng phép toán tensor của PyTorch thay cho `skimage`:

| Phép | Tham số |
| --- | --- |
| Lật ngang | xác suất 0.5 |
| Xoay | góc ngẫu nhiên trong ±10°, `torchvision.transforms.functional.rotate`, bilinear, fill 0 |
| Độ sáng | nhân hệ số ngẫu nhiên 0.9–1.1 |
| Tương phản | `(x - mean) * c + mean`, `c` ngẫu nhiên 0.9–1.1 |

Sau mỗi phép đều clamp về `[0,1]`. Không lật dọc, không affine mạnh, không crop cắt vào khối u — giữ nguyên lập luận ở mục 3.4. Val và test không augment. Hàm `augment_train` bằng skimage ở mục 3.4 vẫn giữ nguyên để minh họa.

`DataLoader`: batch 32, `shuffle=True` cho train, `num_workers=2`, `pin_memory=True`.

### 5.3 Tính tự chứa của mục 4–5

Mục 1–3 chạy ở local, mục 4–5 chạy trên kernel Colab, tức **hai kernel khác nhau**. Vì vậy mục 4–5 không được phụ thuộc biến còn sống từ mục 3 (`df_clean`, `class_weight`, `CLASS_NAMES`, `preprocess_image`). Mục 5.1 tự nạp `.npy` + `.csv`, tự dựng lại `CLASS_NAMES` và tự tính `class_weight`. Mở notebook trên Colab và chạy từ mục 4 phải train được ngay.

`class_weight` phải được **tính lại theo tập train của từng tỷ lệ split** bằng `compute_class_weight("balanced", ...)`, vì mỗi tỷ lệ cho một tập train khác nhau. Dùng chung một bộ weight cho cả 3 tỷ lệ là sai.

## 6. Huấn luyện (mục 5.2–5.3)

- Loss: `CrossEntropyLoss(weight=class_weight_tensor)`. Lớp 7 (*High grade serous*) chỉ có 43 ảnh train ở tỷ lệ 8:1:1, weight balanced là 3.416.
- Optimizer: `AdamW`, `weight_decay=1e-4`.
- Scheduler: giai đoạn 1 lr cố định 1e-3; giai đoạn 2 `CosineAnnealingLR` từ 1e-4 trong 25 epoch.
- Mixed precision: `torch.amp.autocast` + `GradScaler`, nhanh khoảng gấp đôi trên T4 và giảm VRAM. Nếu OOM thì hạ batch xuống 16.
- Early stopping: chỉ áp ở giai đoạn 2, theo **macro-F1 trên val**, patience 7. Giai đoạn 1 luôn chạy đủ 5 epoch. Không dùng accuracy để chọn checkpoint vì dữ liệu lệch lớp nên accuracy bị các lớp đông (0, 2, 5) chi phối.
- Checkpoint: lưu state_dict tốt nhất ra Drive theo tên `resnet50_<split_col>_best.pt` (~90 MB mỗi file).
- Seed 42 cho `random`, `numpy`, `torch`, `torch.cuda`.
- Lịch sử mỗi epoch (train loss, val loss, val accuracy, val macro-F1, lr) lưu ra `train_history_<split_col>.csv`.
- Ba tỷ lệ split được train tuần tự trong một vòng lặp; mỗi tỷ lệ **khởi tạo lại model từ weight ImageNet** với cùng seed.
- Tập test chỉ được dùng một lần duy nhất, trên checkpoint đã chọn theo val, để không rò rỉ thông tin.

Thời gian dự kiến: khoảng 30–40 giây mỗi epoch trên T4, tối đa 30 epoch mỗi tỷ lệ, tổng cho 3 tỷ lệ dưới 1 giờ.

## 7. Đánh giá (mục 5.4–5.5)

Trên tập test của từng tỷ lệ split:

- Accuracy, precision/recall/F1 macro, F1 weighted.
- `classification_report` chi tiết theo 8 lớp.
- Confusion matrix dạng heatmap, có nhãn tên lớp.
- Đường cong train/val loss và val macro-F1 theo epoch, đánh dấu epoch được chọn.

Tổng hợp: một bảng 3 dòng (một dòng mỗi tỷ lệ split) gồm accuracy, macro-F1, weighted-F1, số epoch đã train, lưu ra `results_splits.csv`. Nhận xét phải trả lời được: tỷ lệ chia nào cho kết quả tốt nhất và vì sao, những lớp nào vẫn bị nhầm nhiều, có khớp với dự đoán từ EDA ở mục 2.9 (các lớp minority và các lớp có hình thái gần nhau khó tách) hay không.

Mục 6 kết luận viết sau khi có số liệu, dựa trên `results_splits.csv`.

## 8. Bố cục cell trong notebook

Toàn bộ code nằm trong notebook, không tách file `.py`, giữ đúng kiểu trình bày hiện tại.

| Mục | Nội dung | Chạy ở |
| --- | --- | --- |
| 3.5 | Tạo cache `.npy` + `index.csv`, kiểm tra lại cache | Local |
| 4.0 | Cấu hình môi trường: nhận biết Colab, mount Drive, `device`, seed, pip install `torchinfo` | Colab |
| 4.1 | Markdown: phát biểu bài toán phân loại 8 lớp | — |
| 4.2 | Bảng so sánh 3 backbone (code in số tham số) + lý do chọn ResNet50 | Colab |
| 4.3 | Định nghĩa model ResNet50 + head, `torchinfo.summary` | Colab |
| 4.4 | Hàm đóng băng/mở tầng theo giai đoạn + markdown giải thích | Colab |
| 5.1 | Nạp cache, `Dataset`/`DataLoader`, augment tensor, `class_weight` theo split | Colab |
| 5.2 | Vòng train (train/eval một epoch, AMP, early stopping) | Colab |
| 5.3 | Train cho cả 3 tỷ lệ split | Colab |
| 5.4 | Đánh giá test: report, confusion matrix, đường cong, in và lưu bảng tổng hợp `results_splits.csv` | Colab |
| 5.5 | Markdown nhận xét dựa trên bảng tổng hợp | — |

## 9. Dọn các chỗ còn nhắc phân đoạn

- Mục 1 "Bài toán": "hướng tới phân loại (8 lớp) và phân đoạn vùng u" → chỉ phân loại 8 lớp.
- Mục 1 "Mục tiêu": bỏ `IoU` khỏi danh sách độ đo.
- Mục 3.3: đổi diễn đạt để nói mask chỉ dùng minh họa và đối chiếu trong EDA, không dùng để huấn luyện.
- Mục 4.1: bỏ dòng "(Tùy chọn) Phân đoạn vùng u từ mask nhị phân".
- Mục 5.2: bỏ dòng "Phân đoạn (nếu có): Dice, IoU".
- `docs/ke-hoach-tiep-theo.md`: cập nhật trạng thái mục 4–5 và bỏ nhắc phân đoạn.

Giữ nguyên: mục 2.4, 2.6, 2.7 (EDA dùng mask), hàm `preprocess_mask` và phần minh họa mask ở mục 3.3.

## 10. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
| --- | --- |
| Session Colab đứt giữa lúc train | Checkpoint và history lưu trực tiếp lên Drive sau mỗi epoch cải thiện, train lại được từng tỷ lệ độc lập |
| OOM ở batch 32, ảnh 384×384 | Hạ batch xuống 16; nếu vẫn OOM thì 8 kèm gradient accumulation |
| Overfit do 23.5M tham số nhưng chỉ ~1000 ảnh train | Đóng băng `conv1`/`layer1`/`layer2`, dropout 0.3, weight decay 1e-4, augment nhẹ, early stopping theo macro-F1. Nếu val loss vẫn tách xa train loss ngay từ vài epoch đầu của giai đoạn 2 thì đóng băng thêm `layer3`, chỉ fine-tune `layer4` + `fc` |
| Lớp 7 chỉ 43 ảnh train nên F1 rất thấp | class weight balanced; báo cáo F1 từng lớp thay vì chỉ accuracy; nêu rõ hạn chế ở mục 6 |
| Cache lệch thứ tự với CSV nhãn | Cell 3.5 tự kiểm tra lại một mẫu random và assert `len(npy) == len(csv)` |
| Thống kê BatchNorm trôi ở tầng đã đóng băng | Gọi `.eval()` cho module đóng băng ở mỗi epoch và assert số module đang ở chế độ train |

## 11. Tiêu chí hoàn thành

1. Mục 3.5 chạy được ở local, sinh ra `.npy` + `index.csv` và tự kiểm tra khớp dữ liệu.
2. Mở notebook trên Colab, chạy từ mục 4.0 tới 5.4 không lỗi, không cần chạy lại mục 1–3.
3. Mục 4 in được bảng so sánh 3 backbone và summary của ResNet50.
4. Cả 3 tỷ lệ split đều train xong, có checkpoint, `train_history_<split>.csv` và `results_splits.csv`.
5. Mục 5.4 có confusion matrix, classification report, đường cong loss/F1 cho từng tỷ lệ.
6. Không còn chỗ nào trong notebook nhắc tới phân đoạn như một mục tiêu của đồ án.
