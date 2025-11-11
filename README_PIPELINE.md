# AeroEyes Pipeline - Tài liệu Tổng hợp

Tài liệu này mô tả toàn bộ pipeline end-to-end cho bài toán AeroEyes (Zalo AI Challenge 2025).

## Tổng quan

Pipeline được chia thành 4 giai đoạn chính:

1. **Giai đoạn 1: Data Pipeline** - Preprocessing & Augmentation
2. **Giai đoạn 2: Model Architecture** - Siamese-YOLOv8
3. **Giai đoạn 3: Training Strategy** - Multi-task Loss
4. **Giai đoạn 4: Inference Pipeline** - SORT Tracker & Optimization

## Cấu trúc Dự án

```
AeroEyes/
├── data_pipeline/          # Giai đoạn 1: Data preprocessing & augmentation
│   ├── group_split.py      # Group-based splitting (tránh data leakage)
│   ├── augmentations.py    # Tiling, Copy-Paste, Geometric, Photometric
│   ├── enhanced_prepare_dataset.py
│   └── custom_yolo_dataset.py
│
├── models/                  # Giai đoạn 2: Model architecture
│   ├── siamese_yolo.py     # Siamese-YOLOv8 model
│   └── losses.py           # Multi-task loss (Detection + Triplet)
│
├── tracking/               # Giai đoạn 4: Tracking
│   └── sort_tracker.py     # SORT tracker với Kalman Filter
│
├── inference/              # Giai đoạn 4: Inference pipeline
│   └── pipeline.py        # End-to-end inference pipeline
│
├── utils/                  # Utilities
│   ├── triplet_mining.py  # Hard negative mining
│   └── export_onnx.py     # ONNX/TensorRT export
│
└── observing/              # Dataset
    └── train/
        ├── annotations/
        └── samples/
```

## Hướng dẫn Sử dụng

### Bước 1: Chuẩn bị Dataset

Tạo dataset với Group-Based Splitting và Augmentations để tránh data leakage và tăng cường dữ liệu:

```bash
python data_pipeline/enhanced_prepare_dataset.py --annotations observing/train/annotations/annotations.json --samples observing/train/samples --output dataset_yolo_enhanced --apply-augmentations --train-ratio 0.8
```

Các tùy chọn:
- `--annotations`: Đường dẫn đến file annotations.json
- `--samples`: Thư mục chứa samples (video và reference images)
- `--output`: Thư mục output cho dataset
- `--apply-augmentations`: Áp dụng augmentations (chỉ cho train set)
- `--train-ratio`: Tỷ lệ video cho tập train (mặc định 0.8)
- `--use-all-frames`: Extract tất cả frames thay vì chỉ frames có annotation
- `--sampling-rate`: Tần suất lấy frame khi sử dụng --use-all-frames

### Bước 2: Training Model

Training script đã được triển khai đầy đủ trong `train.py`. Chạy training với:

```bash
python train.py --annotations observing/train/annotations/annotations.json --samples observing/train/samples --output-dir runs/siamese_tracking --epochs 100 --batch-size 8 --lr 1e-4 --img-size 640 --detection-weight 1.0 --similarity-weight 0.5 --triplet-margin 1.0 --train-ratio 0.8
```

Các tham số quan trọng:
- `--epochs`: Số lượng epochs (mặc định 100)
- `--batch-size`: Batch size (mặc định 8)
- `--lr`: Learning rate (mặc định 1e-4)
- `--detection-weight`: Trọng số cho detection loss (mặc định 1.0)
- `--similarity-weight`: Trọng số cho triplet loss (mặc định 0.5)
- `--triplet-margin`: Margin cho triplet loss (mặc định 1.0)
- `--resume`: Đường dẫn đến checkpoint để tiếp tục training

Training script tự động:
- Thực hiện Group-Based Splitting để tránh data leakage
- Sử dụng Multi-task Loss (Detection Loss + Triplet Loss)
- Áp dụng hard negative mining cho triplet loss
- Lưu checkpoint tốt nhất và checkpoint mới nhất
- Hiển thị metrics chi tiết cho cả train và validation

### Bước 3: Inference

```python
from inference.pipeline import AeroEyesInferencePipeline

# Initialize pipeline
pipeline = AeroEyesInferencePipeline(
    model_path="runs/siamese_tracking/best.pt",
    device='cuda'
)

# Initialize với reference images
ref_images = [
    "observing/train/samples/Backpack_0/object_images/img_1.jpg",
    "observing/train/samples/Backpack_0/object_images/img_2.jpg",
    "observing/train/samples/Backpack_0/object_images/img_3.jpg"
]
pipeline.initialize(ref_images)

# Process video
annotations = pipeline.process_video(
    "observing/train/samples/Backpack_0/drone_video.mp4",
    output_path="output_tracked.mp4",
    save_annotations=True
)
```

### Bước 4: Export cho TensorRT (Tùy chọn)

Để tối ưu hóa cho NVIDIA Jetson và đạt > 15 FPS:

```bash
# Export sang ONNX
python utils/export_onnx.py --model runs/siamese_tracking/best.pt --output models/siamese_yolo.onnx --img-size 640 --tensorrt-script

# Compile sang TensorRT (trên máy có TensorRT)
bash export_tensorrt.sh
```

Lưu ý: Custom layers (Attention Pooling, Similarity Head) có thể cần TensorRT plugins. Xem phần "Lưu ý Quan trọng" bên dưới.

## Chi tiết từng Giai đoạn

### Giai đoạn 1: Data Pipeline

**Mục tiêu**: Giải quyết vật thể nhỏ và khoảng cách miền (domain gap)

**Tính năng**:
- Group-Based Splitting (tránh data leakage): Chia dataset theo video_id thay vì frame để đảm bảo tập validation bao gồm các video hoàn toàn chưa từng thấy
- Tiling/Cropping augmentation: Cắt ngẫu nhiên cửa sổ nhỏ hơn từ ảnh gốc để giải quyết vật thể nhỏ (thay thế SAHI)
- Copy-Paste augmentation: Dán vật thể từ frame này sang frame khác để tăng số lượng vật thể nhỏ
- Geometric augmentations: Xoay 90/180/270 độ, lật ngang/dọc để mô phỏng góc nhìn drone
- Photometric augmentations: Điều chỉnh brightness, contrast, gamma correction, thêm noise và motion blur để mô phỏng điều kiện ánh sáng khắc nghiệt

**Xem thêm**: `data_pipeline/README.md`, `data_pipeline/augmentations.py`

### Giai đoạn 2: Model Architecture

**Kiến trúc Siamese-YOLOv8**:
- Shared YOLOv8n backbone (weight sharing): Backbone xử lý video và reference images chia sẻ cùng trọng số (Siamese Network)
- Attention Pooling: Gộp 3 ảnh tham chiếu thành query vector duy nhất bằng Multi-Head Attention, cho phép mô hình học cách gán trọng số cao hơn cho ảnh chất lượng tốt
- Custom detection head: Thay thế detection head tiêu chuẩn của YOLOv8 bằng similarity matching head, tính Cosine Similarity giữa query vector và mọi vị trí trên feature map
- Multi-scale features: Sử dụng 3 scales (P3, P4, P5) từ backbone để phát hiện vật thể ở nhiều kích thước khác nhau

**Xem thêm**: `models/siamese_yolo.py`

### Giai đoạn 3: Training Strategy

**Multi-task Loss**:
- Detection Loss: 
  - CIoU Loss (Complete IoU) cho bounding box regression, tối ưu hóa vị trí (x, y, w, h)
  - Focal Loss cho objectness classification, phân biệt vật thể với hậu cảnh
- Similarity Loss: 
  - Triplet Loss với hard negative mining để thu hẹp khoảng cách miền ground-to-aerial
  - Anchor: Vector đặc trưng từ aerial view (ground-truth bbox)
  - Positive: Query vector từ reference images
  - Negative: Vector đặc trưng từ hard negative objects

Công thức tổng thể: `L_total = w1 * L_detection + w2 * L_similarity`

**Xem thêm**: `models/losses.py`, `utils/triplet_mining.py`

### Giai đoạn 4: Inference Pipeline

**SORT Tracker**:
- Kalman Filter: Làm mượt bounding boxes bị giật và dự đoán vị trí khi phát hiện bị mất
- Hungarian Algorithm: Liên kết detections với tracks dựa trên IoU
- Nội suy: Lấp đầy các khung hình bị bỏ lỡ bằng vị trí dự đoán từ Kalman Filter
- Lọc false positives: Chỉ xuất các tracks đã được xác nhận (min_hits) và loại bỏ tracks quá cũ (max_age)

Pipeline inference tối ưu hóa STIoU bằng cách:
1. Làm mượt bounding boxes (tăng IoU ở tử số)
2. Nội suy khung hình bị bỏ lỡ (giảm frames ở mẫu số)
3. Lọc false positives (giảm frames ở mẫu số)

**Xem thêm**: `tracking/sort_tracker.py`, `inference/pipeline.py`

## Cấu hình

### Model Parameters

```python
model = SiameseYOLOv8(
    model_size='n',          # 'n', 's', 'm', 'l', 'x'
    feature_dim=256,         # Kích thước feature vector
    num_ref_images=3         # Số lượng ảnh tham chiếu
)
```

### Loss Weights

```python
criterion = MultiTaskLoss(
    detection_weight=1.0,    # Weight cho detection loss
    similarity_weight=0.5,    # Weight cho triplet loss
    triplet_margin=1.0       # Margin cho triplet loss
)
```

### Tracker Parameters

```python
tracker = SORTTracker(
    max_age=30,              # Số frames tối đa track tồn tại
    min_hits=3,              # Số lần matched tối thiểu
    iou_threshold=0.3       # Ngưỡng IoU để match
)
```

## Metrics & Evaluation

### STIoU (Spatio-Temporal IoU)

Chỉ số đánh giá chính của bài toán AeroEyes:

```
STIoU = Σ(IoU trên frames chung) / Σ(frames trong union)
```

STIoU không chỉ đơn giản là IoU trung bình. Nó là tổng của IoU trên các khung hình chung (intersection), chia cho tổng số lượng khung hình trong cả ground-truth và dự đoán (union).

Các kịch bản thất bại:
- False Negative: Mô hình phát hiện vật thể ở khung f và f+2, nhưng bỏ lỡ ở khung f+1. Tử số không đổi, nhưng mẫu số tăng thêm 1, làm giảm điểm số.
- Jitter: Bounding box dao động vài pixel giữa các khung hình, làm giảm IoU ở mỗi khung và gây sụt giảm đáng kể cho tử số.
- False Positive: Một dự đoán sai ngẫu nhiên ở một khung hình bất kỳ sẽ tăng mẫu số lên 1 mà không tăng tử số.

SORT tracker tối ưu hóa STIoU bằng cách:
1. Làm mượt bounding boxes (tăng IoU ở tử số)
2. Nội suy khung hình bị bỏ lỡ (giảm frames ở mẫu số)
3. Lọc false positives (giảm frames ở mẫu số)

## Lưu ý Quan trọng

### 1. Custom Layers & TensorRT

Các custom layers (Attention Pooling, Similarity Head) có thể không được hỗ trợ trực tiếp bởi ONNX/TensorRT. Cần:
- Viết custom ONNX operators, hoặc
- Viết TensorRT plugins (C++), hoặc
- Sử dụng PyTorch inference (chậm hơn nhưng đơn giản hơn)

### 2. Triplet Loss Convergence

Triplet Loss nổi tiếng là khó hội tụ. Khuyến nghị:
- Sử dụng pretrained weights (COCO)
- Hard negative mining
- Tinh chỉnh margin cẩn thận
- Warm-up training (bắt đầu với detection loss, sau đó thêm triplet loss)

### 3. Performance trên Jetson

Để đạt > 15 FPS trên Jetson:
- Sử dụng TensorRT FP16
- Tối ưu hóa pipeline (DeepStream)
- Giảm input resolution nếu cần
- Batch processing nếu có thể

## Testing

Kiểm tra các thành phần của pipeline:

```bash
# Test augmentations
python data_pipeline/test_augmentations.py

# Test model architecture
python models/siamese_yolo.py

# Test loss functions
python models/losses.py

# Test SORT tracker
python tracking/sort_tracker.py

# Test inference pipeline
python inference/pipeline.py
```

## TODO / Future Work

- [x] Training script hoàn chỉnh
- [ ] TensorRT plugins cho custom layers (Attention Pooling, Similarity Head)
- [ ] DeepStream integration để đạt > 15 FPS trên Jetson
- [ ] Evaluation script với STIoU metric
- [ ] Hyperparameter tuning (loss weights, learning rate, etc.)
- [ ] Model quantization (INT8) để tăng tốc độ inference
- [ ] Synthetic augmentation (PHẦN 4.3.3) sử dụng Blender/Unreal Engine

## Tài liệu Tham khảo

- YOLOv8: https://github.com/ultralytics/ultralytics
- SORT: "Simple Online and Realtime Tracking" (Bewley et al., 2016)
- Triplet Loss: "FaceNet: A Unified Embedding for Face Recognition" (Schroff et al., 2015)
- TensorRT: https://developer.nvidia.com/tensorrt
- DeepStream: https://developer.nvidia.com/deepstream-sdk

## Cấu trúc File Quan trọng

### Data Pipeline
- `data_pipeline/group_split.py`: Group-based splitting để tránh data leakage
- `data_pipeline/augmentations.py`: Tất cả các augmentations (Tiling, Copy-Paste, Geometric, Photometric)
- `data_pipeline/enhanced_prepare_dataset.py`: Script chuẩn bị dataset với augmentations
- `data_pipeline/custom_yolo_dataset.py`: Custom dataset class cho YOLO format

### Models
- `models/siamese_yolo.py`: Kiến trúc Siamese-YOLOv8 chính
- `models/losses.py`: Multi-task loss functions (CIoU, Focal, Triplet)

### Training & Inference
- `train.py`: Training script hoàn chỉnh với multi-task loss
- `inference/pipeline.py`: End-to-end inference pipeline với SORT tracker
- `tracking/sort_tracker.py`: SORT tracker với Kalman Filter

### Utilities
- `utils/triplet_mining.py`: Hard negative mining cho triplet loss
- `utils/export_onnx.py`: Export model sang ONNX/TensorRT

## Yêu cầu Hệ thống

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (cho GPU training/inference)
- Ultralytics YOLOv8
- OpenCV
- NumPy, SciPy

Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

## Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue hoặc liên hệ team.

---

