# Chỉnh sửa notebook: bỏ so sánh Tab.10, bỏ ghi nguồn bài báo, bỏ tham chiếu mục 5.6

File: `transfer-learning-ovarian-tumor-ultrasound.ipynb`

## Lý do

1. Notebook chỉ dùng một mô hình (ResNet-50), nên không so sánh với Tab.10 của bài báo.
2. Các bảng và câu mô tả chỉ giữ **thành phần và giá trị**, bỏ phần ghi nguồn (mục 5.3, Tab.1, Tab.3, Fig.12, Eq.14, mục 5.1.1, 5.4.1, …).
3. Notebook không có cell nào cho mục 5.6 (nhánh "Model 2" huấn luyện trên ảnh × mask). Mục 5 hiện có đúng 5.1 → 5.5 nên mọi tham chiếu tới 5.6 / Model 2 / `results_roi_mask.csv` đều bị bỏ; không cần đánh số lại các mục còn lại.

Giữ nguyên: dòng giới thiệu bộ dữ liệu "MMOTU — Zhao et al." và các lập luận không mang tính trích nguồn (ví dụ lý do giữ ảnh raw ở mục 3.2).

## Thay đổi theo cell

### Markdown

| Cell | Thay đổi |
| --- | --- |
| 1. Giới thiệu | Mục tiêu: bỏ "rồi đối chiếu với mốc công bố". Đổi tiêu đề "Giao thức bám theo bài báo" → "Giao thức huấn luyện", bỏ câu "Notebook này theo Zhao et al. (2023), mục 5.3 + Tab.3 + Tab.10 + mục 5.4.1", cột "Bài báo quy định" → "Giá trị". Dòng Mô hình → "ResNet-50". Bỏ dòng "Mốc đối chiếu — Tab.10 … 80.17% / 90.19%". |
| 2. EDA (mở đầu) | Lưu ý về mask: chỉ còn "Mục 2.6–2.8 phân tích GT mask chỉ để EDA. Mô hình ResNet-50 ở mục 5 không dùng mask." |
| 2.2 Nhận xét | Bỏ "bài báo mục 5.3"; bỏ "để so sánh được với Tab.10". |
| 2.3 Nhận xét | Bỏ "đúng câu bài báo". |
| 2.4 | Ghi chú mask: bỏ Tab.10 và mục 5.6, chỉ còn "Mô hình ResNet-50 ở mục 5 không dùng mask." |
| 2.4 Nhận xét | Kết luận: "Mask chỉ phục vụ EDA 2.6–2.8. Mô hình ResNet-50 không dùng mask." |
| 2.6 | Bỏ câu về mục 5.6 ở ý Tumor size distribution. |
| 2.6 / 2.7 / 2.8 Nhận xét | Kết luận đổi thành "không đưa vào ResNet-50; chỉ dùng cho EDA", bỏ Tab.10 và mục 5.6. |
| 2.10 Nhận xét | "Bài báo mục 5.4.1 khuyến cáo không can thiệp…" → "Không can thiệp làm lệch phân bố dữ liệu." |
| Tóm tắt đặc trưng | "Nhánh Tab.10: chỉ nhóm 3" → "Baseline mục 3.5 chỉ dùng nhóm 3"; bỏ câu về mục 5.6. Nhóm 4, 5, 6: "EDA + cải tiến 5.6" → "chỉ dùng cho EDA". |
| 3. Tiền xử lý | "Chuẩn bị dữ liệu theo đúng mô tả của bài báo:" → "Chuẩn bị dữ liệu:". |
| 3.4 | Bỏ câu "Mục 5.6 không concat CSV…". |
| 3.4 Nhận xét | Bỏ "Mục 5.6 không dùng file này." |
| 3.5 Nhận xét | View `full`: "baseline này chỉ đo 13 cột intensity, không dùng đặc trưng từ mask." |
| 4.1 | Bỏ đoạn so sánh 7 backbone ở Tab.10, Fig.12 và mốc 80.17% / 90.19%; chỉ còn "Mô hình duy nhất được dùng là ResNet-50." |
| 4.2 | Bỏ câu trích mục 5.3 ("All model are loaded ImageNet-pretrained…"). Chiến lược fine-tune toàn bộ viết lại, không ghi mục 5.3 / Tab.3. |
| 4.3.2 | "không Dropout (bài báo không dùng)" → "không Dropout"; "ngoài phạm vi bài báo cho Task 3" → "ngoài phạm vi đồ án". |
| 4.4 | Tiêu đề bỏ "theo bài báo"; bỏ câu "Bài báo mục 5.3 và Tab.3 quy định…"; bảng cấu hình chỉ còn 2 cột **Thành phần / Giá trị** (bỏ cột Nguồn). Đoạn "Vì sao không giữ hai giai đoạn": bỏ "không thể so sánh trực tiếp với Tab.10". |
| 5. Huấn luyện và Đánh giá | Bỏ câu về mục 5.6; bỏ dòng "5.6 Model 2" trong bảng cấu hình; Metric bỏ "(đúng bộ metric Tab.10)" và "Mốc bài báo"; dòng "Gated fusion / concat CSV trong Tab.10" → "Gated fusion / concat CSV". |
| 5.1 | Thay đoạn "Bài báo ghi gì, code tác giả ghi gì…" và bảng 4 cột bằng bảng **Thành phần / Giá trị** (Train, Test, Không dùng). |
| 5.2 | "Engine rút gọn về đúng những gì bài báo mô tả:" → "Các hàm của engine huấn luyện:". |
| 5.3 | Bỏ "(Tab.1)" và "đó không phải số so với Tab.10". |
| 5.4 | Bỏ "đúng như bài báo báo cáo", "(Tab.10) — đặt cạnh mốc 80.17% / 90.19%", câu trích top-2, "(Fig.12 phải/trái)". |
| 5.5 Nhận xét | Bỏ ý "So với bài báo" (delta_top1/top2); bỏ ý "Model 2 (mục 5.6)"; bỏ "Đúng quan sát mục 5.3 của bài báo"; "bám split bài báo" → "không tách tập val"; Kết luận bỏ Tab.10 và mục 5.6. |
| 6. Kết luận | Bỏ `results_roi_mask.csv`, mốc Tab.10, ý "Model 1 / Model 2"; bỏ "bài báo mô tả ở mục 5.3"; Hạn chế bỏ "chênh lệch so với Tab.10"; Hướng phát triển viết lại ý backbone khác và ý segmentor Task 1 không nhắc Tab.10 / 5.6. |

### Code

| Cell | Thay đổi |
| --- | --- |
| 3.1 split | `print("===== split_paper (đúng Tab.1 bài báo) =====")` → `print("===== split_paper =====")`. |
| 3.2 đọc ảnh | Docstring bỏ "(bài báo mục 5.1.1, 5.4.1)"; tiêu đề hình bỏ "theo bài báo". |
| 3.4 đặc trưng | `print` "Loại khỏi nhánh Tab.10 / 3.5 … 5.6 dùng mask…" → "Loại khỏi baseline 3.5 (phụ thuộc GT mask, N cột):". |
| 3.5 baseline | Xoá dòng `print("Tham chiếu Tab.10 bài báo, ResNet-50 … top1=0.8017 top2=0.9019")`. |
| 4.4 | Comment `# Hằng số đúng theo bài báo (mục 5.3 + Tab.3)` → `# Hằng số huấn luyện`. |
| 5.1 dataset | Bỏ comment nguồn ở `MINORITY_CLASSES`, `LR_BASE`, `MOMENTUM`, `WEIGHT_DECAY`, `POLY_POWER`; xoá hằng `PAPER_RESNET50_TOP1`, `PAPER_RESNET50_TOP2`. |
| 5.2 engine | Bỏ "Tab.10", "bài báo mục 5.3 / Tab.3" trong comment; docstring bỏ "bài báo". |
| 5.4 đánh giá | `print` top-1/top-2 chỉ in giá trị, không in mốc bài báo và độ lệch; tiêu đề hình "tương ứng Fig.12 của bài báo" → "ROC và confusion matrix"; cột `nhom_bai_bao` → `nhom_so_mau`; bỏ cột `paper_top1`, `paper_top2`, `delta_top1`, `delta_top2` khỏi `results_resnet50.csv`; bỏ dòng "ResNet-50 — bài báo Tab.10" khỏi `results_comparison.csv`. |

## Đợt 2: bỏ so sánh với phiên bản cũ của notebook

Notebook chỉ mô tả cấu hình hiện tại, không nhắc "phiên bản trước", "như trước", "đã bỏ", "đã xoá".

| Cell | Thay đổi |
| --- | --- |
| 1. Giới thiệu | Xoá dòng "Toàn bộ khác biệt so với phiên bản trước… `docs/nhat-ky-chinh-sua-theo-bai-bao.md`". |
| 3. Tiền xử lý | Xoá dòng "Nhật ký thay đổi so với phiên bản trước: …". |
| 3.3 | Xoá dòng "Cache cũ `otu2d_448_rgb_u8.npy` / `otu2d_384_u8.npy` … không còn dùng". |
| 4.4 | BatchNorm: "cập nhật `running_mean` / `running_var` bình thường. Không còn phải đồng bộ BN đóng băng như trước." → "cập nhật `running_mean` / `running_var` theo dữ liệu siêu âm." (Đoạn "Vì sao không giữ hai giai đoạn" đã được xoá trước đó.) |
| 5. Huấn luyện và Đánh giá | "Đã bỏ so với phiên bản trước" + bảng "Đã bỏ / Vì" → "Không sử dụng" + bảng "Kỹ thuật / Lý do", lý do viết theo cấu hình hiện tại. |
| 5.2 | Xoá dòng "Các thành phần đã xoá khỏi engine: `GeM`, … `iter_kfold`." |
| 5.3 | Đoạn "Phải `FORCE_RETRAIN=True`… Checkpoint 100 epoch cũ… (75.7% → 72.1%)" → "`FORCE_RETRAIN=False` sẽ nạp lại checkpoint nếu đã có; đặt `FORCE_RETRAIN=True` để train lại từ đầu." |
| 5.4 | Xoá câu "Các file `results_protocol.csv`, … của phiên bản trước không còn được tạo." |
| 5.1 code | Xoá biến `old_square` và gợi ý lỗi "Cache cũ … (vuông 448) không dùng được". |
| 5.3 code | `FORCE_RETRAIN = True  # bắt buộc: … checkpoint cũ lệch phân phối` → `FORCE_RETRAIN = True` (giá trị giữ nguyên). |

## Đợt 3: sửa lỗi `NameError: name 'df_clean' is not defined`

Nguyên nhân: `df_clean` trước đây được tạo trong cell làm sạch dữ liệu (`df_clean = df[df["keep"]].copy()`). Cell đó đã bị bỏ vì không có mẫu nào cần loại, nhưng mục 3.2 và 3.3 vẫn dùng `df_clean`.

| Cell | Thay đổi |
| --- | --- |
| 3.1 code | Cuối cell thêm `df_clean = df.copy()` (giữ toàn bộ 1469 ảnh OTU_2d, vì mục 2.5, 2.10, 2.11 không có mẫu nào cần loại). |
| 3.1 code | Nếu `df` được dựng lại từ file nhãn và thiếu cột `class_name` thì map từ `CLASS_NAMES` (mục 3.3 bắt buộc có cột này). |

## Đợt 4: dọn mục 5.5

Xoá ký tự JSON thừa (`\n",`, `",`, dòng `, tức -`) lọt vào ý Overfitting ở mục 5.5 Nhận xét.

## Lưu ý khi chạy lại

- Output đang lưu của các code cell trên vẫn là bản cũ (còn in Tab.10) cho tới khi chạy lại cell.
- File `runs/results_resnet50.csv` và `runs/per_class_resnet50_paper.csv` sinh ra sau khi chạy lại sẽ không còn cột `paper_*` / `delta_*`, và cột nhóm lớp đổi tên thành `nhom_so_mau`.
