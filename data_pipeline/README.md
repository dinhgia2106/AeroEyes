# Data Pipeline Module - Giai đoạn 1: Preprocessing & Augmentation

Module này triển khai **Giai đoạn 1** của pipeline AeroEyes theo tài liệu chiến lược, tập trung vào việc chuẩn bị và tăng cường dữ liệu để giải quyết:

1. **Rò rỉ Dữ liệu (Data Leakage)**: Group-Based Splitting
2. **Vật thể nhỏ (Small Objects)**: Tiling/Cropping và Copy-Paste augmentation
3. **Khoảng cách Miền (Domain Gap)**: Geometric và Photometric augmentations

## 📁 Cấu trúc Module

```
data_pipeline/
├── __init__.py                    # Module exports
├── group_split.py                 # Group-Based Splitting để tránh data leakage
├── augmentations.py               # Tất cả các augmentation techniques
├── enhanced_prepare_dataset.py    # Script chuẩn bị dataset với augmentations
├── custom_yolo_dataset.py         # Custom dataset với augmentations on-the-fly
├── test_augmentations.py          # Script test các augmentations
└── README.md                      # Tài liệu này
```

## 🚀 Sử dụng

### 1. Group-Based Splitting

Thay vì chia ngẫu nhiên các frames, chúng ta chia theo `video_id` để tránh data leakage:

```python
from data_pipeline.group_split import group_based_split

train_video_ids, val_video_ids = group_based_split(
    annotations_path="observing/train/annotations/annotations.json",
    train_ratio=0.8,
    random_seed=42
)
```

### 2. Chuẩn bị Dataset với Augmentations

Sử dụng script `enhanced_prepare_dataset.py` để tạo dataset với augmentations:

```bash
# Tạo dataset không có augmentations (chỉ group-based splitting)
python data_pipeline/enhanced_prepare_dataset.py \
    --annotations observing/train/annotations/annotations.json \
    --samples observing/train/samples \
    --output dataset_yolo_enhanced

# Tạo dataset với augmentations (chỉ áp dụng cho train set)
python data_pipeline/enhanced_prepare_dataset.py \
    --annotations observing/train/annotations/annotations.json \
    --samples observing/train/samples \
    --output dataset_yolo_enhanced \
    --apply-augmentations \
    --train-ratio 0.8
```

### 3. Sử dụng Custom Dataset với Augmentations On-the-Fly

Để áp dụng augmentations trong quá trình training (đa dạng hóa hơn):

```python
from data_pipeline.custom_yolo_dataset import AugmentedYOLODataset

# Tạo dataset cho training
train_dataset = AugmentedYOLODataset(
    images_dir="dataset_yolo/images/train",
    labels_dir="dataset_yolo/labels/train",
    img_size=(640, 640),
    apply_augmentations=True,
    tiling_prob=0.5,
    geometric_prob=0.5,
    photometric_prob=0.7
)

# Dataset cho validation (không có augmentations)
val_dataset = AugmentedYOLODataset(
    images_dir="dataset_yolo/images/val",
    labels_dir="dataset_yolo/labels/val",
    img_size=(640, 640),
    apply_augmentations=False
)
```

## 🔧 Các Augmentation Techniques

### 1. Tiling/Cropping Augmentation

**Mục đích**: Giải quyết vật thể nhỏ bằng cách "zoom kỹ thuật số"

**Cách hoạt động**: Cắt ngẫu nhiên một cửa sổ nhỏ hơn từ ảnh gốc, buộc mô hình học các đặc trưng chi tiết của vật thể nhỏ.

```python
from data_pipeline.augmentations import TilingAugmentation

tiling_aug = TilingAugmentation(
    crop_size=(640, 640),
    min_scale=0.5,
    max_scale=1.0,
    prob=0.5
)
```

### 2. Copy-Paste Augmentation

**Mục đích**: Tăng số lượng vật thể nhỏ và hiếm

**Cách hoạt động**: Cắt một vật thể từ một frame và dán vào frame khác với các biến đổi (scale, rotation, brightness).

```python
from data_pipeline.augmentations import CopyPasteAugmentation

copy_paste_aug = CopyPasteAugmentation(
    prob=0.3,
    max_objects=2,
    scale_range=(0.8, 1.2),
    rotation_range=(-15, 15)
)
```

### 3. Geometric Augmentations

**Mục đích**: Mô phỏng các góc nhìn khác nhau của drone (không có hướng "lên" cố định)

**Các phép biến đổi**:
- Rotation: 90°, 180°, 270°
- Horizontal Flip
- Vertical Flip

```python
from data_pipeline.augmentations import GeometricAugmentation

geometric_aug = GeometricAugmentation(
    rotation_angles=[90, 180, 270],
    flip_horizontal=True,
    flip_vertical=True,
    prob=0.5
)
```

### 4. Photometric Augmentations

**Mục đích**: Mô phỏng điều kiện ánh sáng khắc nghiệt (nắng gắt, thiếu sáng)

**Các phép biến đổi**:
- Brightness/Contrast Jitter
- Gamma Correction (gamma < 1: nắng gắt, gamma > 1: thiếu sáng)
- Gaussian Noise
- Motion Blur

```python
from data_pipeline.augmentations import PhotometricAugmentation

photometric_aug = PhotometricAugmentation(
    brightness_range=(-30, 30),
    contrast_range=(0.8, 1.2),
    gamma_range=(0.7, 1.3),
    noise_std=5.0,
    blur_prob=0.3,
    prob=0.7
)
```

## 🧪 Testing

Chạy script test để kiểm tra các augmentations:

```bash
python data_pipeline/test_augmentations.py
```

Script này sẽ:
- Test Tiling augmentation
- Test Geometric augmentation
- Test Photometric augmentation
- Test Group-Based Splitting

Và tạo các file ảnh output để kiểm tra trực quan.

## 📊 So sánh với Pipeline Cũ

| Tính năng | `prepare_dataset.py` (cũ) | `enhanced_prepare_dataset.py` (mới) |
|-----------|---------------------------|-------------------------------------|
| Splitting | Chia theo thứ tự video | **Group-Based Splitting** (tránh data leakage) |
| Augmentations | Không có | **Tiling, Copy-Paste, Geometric, Photometric** |
| Copy-Paste | Không có | ✅ Có |
| On-the-fly Aug | Không có | ✅ Có (qua `custom_yolo_dataset.py`) |

## 🎯 Lưu ý Quan trọng

1. **Group-Based Splitting là bắt buộc**: Chia ngẫu nhiên frames sẽ gây ra data leakage nghiêm trọng.

2. **Augmentations chỉ cho Train Set**: Validation set không bao giờ được augment để đánh giá chính xác.

3. **Copy-Paste yêu cầu Memory**: Copy-Paste augmentation cần load frames vào memory, có thể tốn RAM với dataset lớn.

4. **On-the-Fly vs Pre-Process**: 
   - **Pre-process** (trong `enhanced_prepare_dataset.py`): Augmentations được áp dụng một lần khi tạo dataset
   - **On-the-fly** (trong `custom_yolo_dataset.py`): Augmentations được áp dụng mỗi epoch, đa dạng hơn nhưng chậm hơn

## 📚 Tài liệu Tham khảo

Các augmentation techniques được triển khai dựa trên:
- **Tiling**: "The Power of Tiling for Small Object Detection" (SAHI paper)
- **Copy-Paste**: "Simple Copy-Paste is a Strong Data Augmentation Method"
- **Geometric/Photometric**: Best practices cho aerial/drone data

## 🔄 Tích hợp với YOLOv8

Để sử dụng với YOLOv8 training, bạn có thể:

1. **Option 1**: Sử dụng dataset đã được pre-process với augmentations
   ```bash
   yolo train data=dataset_yolo_enhanced/data.yaml model=yolov8n.pt
   ```

2. **Option 2**: Tích hợp `AugmentedYOLODataset` vào training loop của YOLOv8 (cần custom training script)

## ✅ Checklist Triển khai

- [x] Group-Based Splitting
- [x] Tiling/Cropping Augmentation
- [x] Copy-Paste Augmentation
- [x] Geometric Augmentations (Rotation, Flips)
- [x] Photometric Augmentations (Brightness, Contrast, Gamma, Noise, Blur)
- [x] Enhanced Dataset Preparation Script
- [x] Custom Dataset với On-the-Fly Augmentations
- [x] Test Scripts
- [x] Documentation


