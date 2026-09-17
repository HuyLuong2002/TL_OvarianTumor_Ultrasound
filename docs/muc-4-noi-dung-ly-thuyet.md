# Mục 4 "Xây dựng mô hình" — nội dung lý thuyết để chọn lọc

Tài liệu liệt kê và viết sẵn phần lý thuyết mục 4 để bạn cắt lọc vào notebook. **Công thức dùng LaTeX** (`$...$`, `$$...$$`) — dán vào **markdown cell** của Jupyter/Colab rồi chạy cell để hiển thị đẹp. Trình xem file `.md` trong IDE có thể không render math; notebook thì render được.

Mục 4 chỉ nói **kiến trúc và cơ sở lý thuyết**; loss, optimizer, lịch train thuộc mục 5.

---

## A. Danh sách nội dung cần trình bày

| Mục | Nội dung | Vai trò | Dạng |
| --- | --- | --- | --- |
| 4.1 | Phát biểu bài toán: đầu vào, đầu ra, giả thiết | Bắt buộc | Markdown |
| 4.2 | Transfer learning: định nghĩa, vì sao hiệu quả, ba chiến lược, chọn chiến lược nào | Bắt buộc | Markdown |
| 4.3 | Kiến trúc ResNet: degradation, residual learning, bottleneck | Bắt buộc | Markdown |
| 4.4 | Cấu hình ResNet50 theo tầng, feature map 384×384, phân bố tham số | Bắt buộc | Markdown + bảng |
| 4.5 | Ảnh xám → 3 kênh, chuẩn hóa ImageNet, head 8 lớp | Bắt buộc | Markdown |
| 4.6 | Fine-tune hai giai đoạn, BatchNorm khi đóng băng | Bắt buộc | Markdown + bảng |
| 4.7 | So sánh ResNet50 / EfficientNet-B0 / DenseNet121, lý do chọn | Bắt buộc | Markdown + bảng |
| 4.8 | Sơ đồ kiến trúc (`torchsummary`) + bảng tham số + shape từng stage | Bắt buộc | Code |
| 4.9 | Chi phí tính toán: FLOPs, VRAM, thời gian/epoch | Tùy chọn | Markdown |
| 4.10 | Tài liệu tham khảo | Tùy chọn | Markdown |

---

## B. Nội dung chi tiết

### 4.1 Bài toán mô hình hóa

Bài toán là **phân loại đơn nhãn, đa lớp** (single-label multi-class). Mô hình là một hàm tham số hóa:

$$
f_\theta: \mathbb{R}^{384 \times 384} \rightarrow \mathbb{R}^{8}
$$

nhận một ảnh siêu âm xám đã chuẩn hóa ở mục 3.3 và trả về vector logit 8 chiều $\mathbf{z} = (z_0, z_1, \ldots, z_7)$. Xác suất lớp thu được qua softmax:

$$
p_k = \frac{\exp(z_k)}{\sum_{j=0}^{7} \exp(z_j)}
$$

$$
\hat{y} = \arg\max_k p_k
$$

trong đó $\theta$ là toàn bộ tham số mạng, $z_k$ là logit lớp $k$, $p_k$ là xác suất lớp $k$ và $\hat{y}$ là nhãn dự đoán.

**Đầu vào:** ảnh xám một kênh, 384×384, giá trị trong $[0,1]$ sau pad vuông, resize, CLAHE và min-max.

**Đầu ra:** một trong 8 lớp bệnh lý (mục 1).

**Giả thiết:** mỗi ảnh thuộc đúng một lớp; thông tin quyết định nằm trong một ảnh 2D (không chuỗi ảnh, không thông tin lâm sàng kèm theo). Mask nhị phân **không** dùng làm đầu vào hay nhãn huấn luyện, chỉ phục vụ EDA (mục 2).

### 4.2 Cơ sở lý thuyết transfer learning

**Định nghĩa.** Một *domain* gồm không gian đặc trưng và phân phối biên $\mathcal{D} = \{\mathcal{X}, P(X)\}$; một *task* gồm không gian nhãn và hàm dự đoán $\mathcal{T} = \{\mathcal{Y}, f(\cdot)\}$. Transfer learning dùng kiến thức từ domain nguồn $\mathcal{D}_s$, task nguồn $\mathcal{T}_s$ để cải thiện học trên domain đích $\mathcal{D}_t$, task đích $\mathcal{T}_t$ khi $\mathcal{D}_s \neq \mathcal{D}_t$ hoặc $\mathcal{T}_s \neq \mathcal{T}_t$ (Pan & Yang, 2010).

| | Nguồn | Đích |
| --- | --- | --- |
| Dữ liệu | ImageNet (~1.28M ảnh RGB) | OTU_2d (1469 ảnh siêu âm xám) |
| Số lớp | 1000 | 8 |
| Đặc điểm | Ảnh màu, tương phản cao | Xám, speckle, biên mờ |

**Vì sao hiệu quả.** CNN học đặc trưng theo tầng: tầng đầu (cạnh, texture) ít phụ thuộc loại ảnh; tầng sâu mang ngữ nghĩa task nguồn. Yosinski et al. (2014): tính chuyển giao giảm theo độ sâu; domain xa ImageNet thì fine-tune tầng sâu thường lợi hơn chỉ thay classifier. Với train ~1175 ảnh (8:1:1), train from scratch trên ResNet50 (~23.5M tham số) dễ overfit; khởi tạo ImageNet giảm epoch và phương sai.

**Ba chiến lược**

| Chiến lược | Cách làm | Phù hợp khi |
| --- | --- | --- |
| Feature extraction | Đóng băng backbone, chỉ train head | Dữ liệu đích rất ít, gần ImageNet |
| Fine-tune một phần | Đóng băng tầng đầu, train tầng sâu + head | Ít–vừa mẫu, domain khác |
| Fine-tune toàn bộ | Cập nhật mọi tham số, lr nhỏ | Rất nhiều ảnh đích |

Nếu $\theta = \theta_{\text{freeze}} \cup \theta_{\text{train}}$:

$$
\theta_{\text{train}} \leftarrow \theta_{\text{train}} - \eta \nabla_{\theta_{\text{train}}} \mathcal{L}, \qquad \theta_{\text{freeze}} = \text{const}
$$

**Chiến lược chọn:** fine-tune một phần (mục 4.6). Cảnh báo *negative transfer*: lr lớn + fine-tune toàn bộ trên tập nhỏ có thể phá weight pretrained — cần warm-up head và lr nhỏ ở giai đoạn 2.

### 4.3 Kiến trúc ResNet

**Degradation (He et al., 2016).** Mạng CNN thông thường sâu hơn có thể làm **sai số train tăng** (không phải overfit thuần). Residual block học phần dư thay vì ánh xạ trực tiếp $\mathcal{H}(x)$:

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

Khi ánh xạ gần identity, $\mathcal{F} \approx 0$ dễ học hơn. Shortcut giúp gradient lan truyền (nhánh cộng có đạo hàm 1). Khi đổi số kênh/kích thước dùng *projection shortcut* (conv 1×1 + stride).

**Bottleneck (ResNet50).** Mỗi khối: 1×1 giảm kênh → 3×3 → 1×1 mở rộng ×4. ResNet50: 4 stage với [3, 4, 6, 3] khối bottleneck.

### 4.4 Cấu hình chi tiết ResNet50 với ảnh 384×384

| Tầng | Cấu hình | Kích thước đầu ra | Số tham số |
| --- | --- | --- | --- |
| Đầu vào | 3 kênh | 3 × 384 × 384 | 0 |
| `conv1` | 7×7, 64, stride 2 | 64 × 192 × 192 | 9,408 |
| `bn1` + ReLU | | 64 × 192 × 192 | 128 |
| `maxpool` | 3×3, stride 2 | 64 × 96 × 96 | 0 |
| `layer1` | 3 bottleneck, 256 ch | 256 × 96 × 96 | 215,808 |
| `layer2` | 4 bottleneck, 512 ch, stride 2 | 512 × 48 × 48 | 1,219,584 |
| `layer3` | 6 bottleneck, 1024 ch | 1024 × 24 × 24 | 7,098,368 |
| `layer4` | 3 bottleneck, 2048 ch | 2048 × 12 × 12 | 14,964,736 |
| `avgpool` | global average pooling | 2048 | 0 |
| `fc` (đã thay) | Dropout 0.3 + Linear(2048→8) | 8 | 16,392 |
| **Tổng** | | | **23,524,424** |

**Nhận xét:** ảnh 384×384 lớn hơn 224×224 huấn luyện gốc nhưng vẫn chạy được nhờ global average pooling; feature map cuối 12×12 (thay vì 7×7) giữ chi tiết mịn hơn — hữu ích cho siêu âm. `layer3` + `layer4` chiếm ~93.8% tham số → chiến lược đóng băng tầng đầu (mục 4.6) vẫn để ~94% tham số được fine-tune.

### 4.5 Điều chỉnh cho bài toán

**Ảnh xám → 3 kênh.** `conv1` nhận 3 kênh; nhân bản ảnh xám thành 3 kênh giống nhau để giữ nguyên weight ImageNet và mean/std ImageNet (thay vì gộp weight conv theo kênh).

**Chuẩn hóa.** Sau $[0,1]$ từ mục 3.3, áp mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]` — khớp phân phối đầu vào lúc pretrain.

**Head.** Thay `fc` gốc bằng `Dropout(0.3)` + `Linear(2048, 8)`. Dropout chống overfit khi vector đặc trưng 2048 chiều nhưng train chỉ ~1000 ảnh.

**Weight.** `ResNet50_Weights.IMAGENET1K_V2` (top-1 ~80.86% vs V1 ~76.13%), cùng kiến trúc.

### 4.6 Chiến lược fine-tune hai giai đoạn

**Giai đoạn 1 — warm-up head.** Đóng băng backbone, chỉ train `fc`, 5 epoch, lr $10^{-3}$.

**Giai đoạn 2.** Mở `layer3`, `layer4`, `fc`; đóng băng `conv1`, `bn1`, `layer1`, `layer2`. 25 epoch, lr $10^{-4}$, cosine decay. Early stopping theo macro-F1 trên val (mục 5).

| Thành phần | Giai đoạn 1 | Giai đoạn 2 |
| --- | --- | --- |
| `conv1`, `bn1`, `layer1`, `layer2` | đóng băng | đóng băng |
| `layer3`, `layer4` | đóng băng | **train** |
| `fc` | **train** | **train** |

**BatchNorm khi đóng băng:** chỉ `requires_grad=False` **không đủ** — buffer `running_mean`/`running_var` vẫn cập nhật ở mode train. Module đóng băng phải `.eval()` mỗi epoch train; nên assert số BN đang ở train mode.

**Dự phòng overfit:** nếu val loss tách train loss sớm ở giai đoạn 2, đóng băng thêm `layer3`, chỉ fine-tune `layer4` + `fc`.

### 4.7 So sánh với hai kiến trúc khác

| Kiến trúc | Ý tưởng | Tham số (head 1000 lớp) | Chiều feature | ImageNet top-1 (torchvision) |
| --- | --- | --- | --- | --- |
| **ResNet50** | Residual + bottleneck | 25.56M | 2048 | 76.13% (V1) / **80.86% (V2)** |
| DenseNet121 | Dense connection | 7.98M | 1024 | 74.43% |
| EfficientNet-B0 | MBConv + compound scaling | 5.29M | 1280 | 77.69% |

**Chọn ResNet50:** đối chiếu được baseline MMOTU (họ ResNet); dễ trình bày lý thuyết residual; có weight V2 mạnh.

**Đánh đổi:** 23.5M tham số vs ~1000 ảnh train → overfit cao hơn DenseNet/EfficientNet → cần đóng băng tầng đầu, dropout, weight decay, augment, early stopping.

### 4.8 Sơ đồ kiến trúc

Phần code ở mục C; output mẫu ở mục D.

### 4.9 Chi phí tính toán

ResNet50 ~4.09 GFLOPs @ 224×224; scale theo diện tích ảnh: $\approx 4.09 \times (384/224)^2 \approx 12$ GFLOPs/ảnh. Trên T4, batch 32, mixed precision: ~30–40 s/epoch (train+val) với ~1175 ảnh train; 3 split < ~1 giờ tổng.

`torchsummary` (1 ảnh, float32): activation forward/backward ~842 MB — batch 32 float32 có thể vượt VRAM T4; **mixed precision gần như bắt buộc**; nếu OOM hạ batch 16.

### 4.10 Tài liệu tham khảo

1. K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
2. J. Yosinski et al. *How Transferable Are Features in Deep Neural Networks?* NeurIPS 2014.
3. S. J. Pan, Q. Yang. *A Survey on Transfer Learning.* IEEE TKDE, 2010.
4. S. Ioffe, C. Szegedy. *Batch Normalization.* ICML 2015.
5. N. Srivastava et al. *Dropout.* JMLR, 2014.
6. J. Deng et al. *ImageNet.* CVPR 2009.
7. G. Huang et al. *Densely Connected Convolutional Networks.* CVPR 2017.
8. M. Tan, Q. V. Le. *EfficientNet.* ICML 2019.
9. Q. Zhao et al. *MMOTU: A Multi-Modality Ovarian Tumor Ultrasound Image Dataset.*

---

## C. Code cho mục 4.8 — sơ đồ kiến trúc

### Cell 1: định nghĩa model và `torchsummary`

```python
# Nếu thiếu: pip install torchsummary
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 8
IMG_SIZE = 384
DROPOUT_P = 0.3


def build_resnet50(num_classes=NUM_CLASSES, dropout_p=DROPOUT_P):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_resnet50().to(device)

print("Thiết bị:", device)
summary(model, input_size=(3, IMG_SIZE, IMG_SIZE), batch_size=-1, device=device)
```

### Cell 2: bảng tham số theo tầng

```python
import pandas as pd

STAGES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
rows = []
for stage in STAGES:
    module = getattr(model, stage)
    rows.append({"Thành phần": stage, "Số tham số": sum(p.numel() for p in module.parameters())})

param_df = pd.DataFrame(rows)
param_df["Tỷ lệ (%)"] = (100 * param_df["Số tham số"] / param_df["Số tham số"].sum()).round(2)
print(param_df.to_string(index=False))
print(f"\nTổng tham số: {int(param_df['Số tham số'].sum()):,}")
deep = param_df["Thành phần"].isin(["layer3", "layer4"])
print(f"layer3 + layer4: {int(param_df.loc[deep, 'Số tham số'].sum()):,}")
```

### Cell 3: kích thước feature map

```python
model.eval()
x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

with torch.no_grad():
    print(f"{'input':10s} {tuple(x.shape)}")
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    print(f"{'stem':10s} {tuple(x.shape)}")
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        x = getattr(model, stage)(x)
        print(f"{stage:10s} {tuple(x.shape)}")
    x = torch.flatten(model.avgpool(x), 1)
    print(f"{'avgpool':10s} {tuple(x.shape)}")
    print(f"{'fc':10s} {tuple(model.fc(x).shape)}")
```

---

## D. Output mẫu (torch 2.14, ResNet50 V2, ảnh 384×384)

**Cell 1 — tóm tắt cuối bảng `torchsummary`:**

```text
Total params: 23,524,424
Trainable params: 23,524,424
Input size (MB): 1.69
Forward/backward pass size (MB): 842.09
Params size (MB): 89.74
Estimated Total Size (MB): 933.52
```

**Cell 2:**

```text
Thành phần  Số tham số  Tỷ lệ (%)
     conv1        9408       0.04
       bn1         128       0.00
    layer1      215808       0.92
    layer2     1219584       5.18
    layer3     7098368      30.17
    layer4    14964736      63.61
        fc       16392       0.07

Tổng tham số: 23,524,424
layer3 + layer4: 22,063,104
```

**Cell 3:**

```text
input      (1, 3, 384, 384)
stem       (1, 64, 96, 96)
layer1     (1, 256, 96, 96)
layer2     (1, 512, 48, 48)
layer3     (1, 1024, 24, 24)
layer4     (1, 2048, 12, 12)
avgpool    (1, 2048)
fc         (1, 8)
```
