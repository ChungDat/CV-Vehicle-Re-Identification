# Vehicle Re-Identification (ReID)
## Cấu trúc mã nguồn

```text
propose/
├── datasets/                      # Thư mục chứa dữ liệu
│   ├── VeRi776/                   # Dataset VeRi776
│   │   ├── image_query/           # Tập ảnh truy vấn
│   │   ├── image_test/            # Tập ảnh gallery
│   │   ├── image_train/           # Tập ảnh huấn luyện
│   │   └── ...
│   └── VRIC/                      # Dataset VRIC
│       ├── gallery_images/        # Tập ảnh gallery
│       ├── probe_images/          # Tập ảnh truy vấn
│       ├── train_images/          # Tập ảnh huấn luyện
│       ├── vric_gallery.txt       # Nhãn của ảnh gallery
│       ├── vric_probe.txt         # Nhãn của ảnh query
│       ├── vric_train.txt         # Nhãn của ảnh train
│       └── README.md              # Tài liệu hướng dẫn về dataset VRIC
├── outputs/                       # Thư mục lưu model checkpoints, logs và kết quả
│   ├── baseline/
│   │   └── YYYY_MM_DD_HH_MM_SS/
│   │       ├── checkpoint/        # Chứa weights (best_model.pth)
│   │       ├── logs/              # Chứa file log quá trình train
│   │       ├── plots/             # Biểu đồ training metrics
│   │       ├── test_result/       # Kết quả đánh giá và truy vấn
│   │       ├── history.json       # Json file lưu thông số train
│   │       └── metrics.png        # Hình ảnh biểu đồ training metrics
│   ├── baseline_with_BOT/
│   ├── baseline_with_CBAM/
│   └── BOT_with_CBAM/
├── scripts/                       # Shell script
│   ├── train_baseline_cosine.sh
│   ├── train_baseline_euclid.sh
│   └── ...
├── config.py                      # Cấu hình siêu tham số, đường dẫn và hệ thống
├── load_dataset.py                # Load dataset (VeRi776, VRIC)
├── loss.py                        # Định nghĩa các hàm loss
├── model.py                       # Định nghĩa kiến trúc mô hình
├── pyproject.toml                 # File cấu hình thư viện cho uv
├── retrieve_image.py              # Script để truy vấn ảnh
├── sampler.py                     # Batch sampler
├── test_model.py                  # Đánh giá mô hình
├── train_model.py                 # Vòng lặp huấn luyện chính
├── train_one_epoch.py             # Logic huấn luyện cho 1 epoch
├── utils.py                       # Các hàm tiện ích
└── README.md                      # Tài liệu hướng dẫn sử dụng (file này)
```

---

## Hướng dẫn trên Local

### 1. Chuẩn bị dữ liệu
Tải các dataset và đặt vào thư mục `datasets/` theo đúng cấu trúc
* **VeRi776**: [Kaggle](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)
* **VRIC**: [Github](https://qmul-vric.github.io/)

### 2. Thiết lập môi trường
Mã nguồn sử dụng `uv` để quản lý môi trường. Chạy lệnh để cài đặt các thư viện và kích hoạt môi trường:
```bash
uv sync --active
```

### 3. Huấn luyện mô hình
Sử dụng bash scripts có sẵn trong thư mục `scripts/`:
```bash
bash ./scripts/train_baseline_cosine.sh
```

Hoặc chạy trực tiếp bằng Python để tùy chỉnh tham số:
```bash
python train_model.py \
  --model-type baseline \
  --total-epochs 60 \
  --batch-size 32 \
  --num-instances 8 \
  --warmup-epochs 0 \
  --random-erasing False \
  --label-smoothing False \
  --distance-mode cosine \
  --dataset-name VeRi776
```

**Các tham số:**
* `--model-type`: `baseline`, `baseline_with_BOT`, `baseline_with_CBAM`, `BOT_with_CBAM`.
* `--dataset-name`: `VeRi776` hoặc `VRIC` (mặc định: `VeRi776`).
* `--distance-mode`: `cosine` hoặc `euclid` (mặc định: `cosine`).
* `--total-epochs`: Tổng số epoch (mặc định: `60`).
* `--batch-size`: Kích thước batch (mặc định: `32`).
* `--num-instances`: Số ảnh (instance) của mỗi xe trong 1 batch (mặc định: `8`).
* `--warmup-epochs`: Số epoch áp dụng warmup learning rate (Chỉ áp dụng cho cấu hình có BOT) (mặc định: `10`).
* `--random-erasing`: Sử dụng Random Erasing (Chỉ áp dụng cho cấu hình có BOT) (mặc định: `False`).
* `--label-smoothing`: Sử dụng Label Smoothing (Chỉ áp dụng cho cấu hình có BOT) (mặc định: `False`).

Kết quả huấn luyện được lưu tại `outputs/<model_type>/<timestamp>/`.

### 4. Đánh giá mô hình
```bash
python test_model.py \
  --dataset-name VeRi776 \
  --model-type baseline \
  --model-path ./outputs/baseline/YYYY_MM_DD_HH_MM_SS/checkpoint/best_model.pth
```
*Nếu không truyền `--model-path`, script sẽ sử dụng checkpoint được huấn luyện gần nhất.*

Kết quả đánh giá được lưu tại `outputs/<model_type>/<timestamp>/test_result/`.

### 5. Thực hiện truy vấn ảnh
```bash
python retrieve_image.py \
  --dataset-name VeRi776 \
  --model-type baseline \
  --model-path ./outputs/baseline/YYYY_MM_DD_HH_MM_SS/checkpoint/best_model.pth \
  --img-name 0002_c002_00030600_0.jpg \
  --top-k 10
```

**Các tham số:**
* `--img-name`: Tên của ảnh query muốn tìm kiếm. (Chỉ cần tên ảnh, script sẽ thực hiện tìm trong 2 dataset tương ứng)
* `--top-k`: Số lượng ảnh kết quả trả về (mặc định: `10`).

Kết quả truy vấn được lưu tại `outputs/<model_type>/<timestamp>/test_result/`.

---

## Hướng dẫn trên Kaggle

### 1. Chuẩn bị Notebook
* Bật GPU.
* Thêm các dataset VeRi776 và VRIC vào Input của Kaggle.

### 2. Clone mã nguồn và cài đặt môi trường
```bash
!git clone https://github.com/ChungDat/CV-Vehicle-Re-Identification.git
%cd CV-Vehicle-Re-Identification/propose
%mkdir datasets

# Cài đặt uv và các dependency
!pip install tensorboardX

%cp -r /kaggle/input/datasets/abhyudaya12/veri-vehicle-re-identification-dataset/VeRi /kaggle/working/CV-Vehicle-Re-Identification/propose/datasets/VeRi776 # dataset VeRi776 có thể được import trực tiếp từ Kaggle Datasets

%cp -r /kaggle/input/datasets/datchungt/vricdata /kaggle/working/CV-Vehicle-Re-Identification/propose/datasets/VRIC # dataset VRIC cần được upload thủ công lên Kaggle và thay thế đường dẫn phù hợp
```
### 3. Huấn luyện model

```bash
!bash scripts/train_baseline_cosine.sh
# Hoặc các mô hình khác:
# !bash scripts/train_baseline_with_BOT_cosine.sh
# !bash scripts/train_baseline_with_CBAM_cosine.sh
# !bash scripts/train_BOT_with_CBAM_cosine.sh

# Đánh giá sau khi huấn luyện
!python test_model.py --dataset-name VeRi776
!python test_model.py --dataset-name VRIC
```

### 4. Lưu output
```bash
!mv -r /kaggle/working/CV-Vehicle-Re-Identification/propose/outputs/ /kaggle/working/
!rm -rf /kaggle/working/CV-Vehicle-Re-Identification/
```
*Thời gian huấn luyện mô hình với cấu hình trong `scripts/` khoảng 8-12 tiếng.*

## Checkpoint của mô hình đã huấn luyện
Checkpoint của các cấu hình được huấn luyện trên Kaggle với VeRi776 và scripts sử dụng cosine được lưu tại: [Google Drive](https://drive.google.com/drive/folders/1H0bFWvNINiSciRjCS239_WXoWHAyW9cT?usp=sharing)