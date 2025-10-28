# Drone Object Detection - YOLOv8 Template Matching Approach

Hệ thống phát hiện vật thể trong video drone sử dụng YOLOv8, được tối ưu để chạy trên phần cứng Jetson với tốc độ >15 FPS.

## Giới thiệu

Dự án này thực hiện việc tìm kiếm vật thể trong video drone thông qua việc sử dụng 3 ảnh tham chiếu của vật thể. Mô hình được train bằng cách extract các frames từ video với annotations và tạo dataset theo format YOLO.

## Cấu trúc dự án

```
.
├── observing/
│   └── train/
│       ├── annotations/
│       │   └── annotations.json      # Ground truth annotations
│       └── samples/                   # Training video samples
├── public_test/
│   └── public_test/
│       └── samples/                    # Test video samples
├── dataset_yolo/                       # Generated dataset (sau khi chạy prepare_dataset.py)
├── requirements.txt                    # Python dependencies
├── prepare_dataset.py                 # Script chuẩn bị dataset
├── train.py                           # Script train model
├── test.py                            # Script inference và tạo predictions
├── utils.py                            # Utility functions
├── visualize_annotations.py            # Script visualize annotations
├── run_all.py                         # Script chạy toàn bộ pipeline
└── README.md                          # File này
```

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dataset

Extract frames từ video và tạo dataset theo format YOLO:

```bash
python prepare_dataset.py --annotations observing/train/annotations/annotations.json --samples observing/train/samples --output dataset_yolo
```

**Các options:**
- `--annotations`: Path đến annotations.json
- `--samples`: Path đến samples directory
- `--output`: Directory output cho dataset
- `--use-all-frames`: Extract tất cả frames thay vì chỉ frames có annotation
- `--sampling-rate`: Frame sampling rate khi dùng --use-all-frames (default: 10)

### 3. Train model

Train YOLOv8n model (nhẹ nhất, phù hợp cho Jetson):

```bash
python train.py --data dataset_yolo/data.yaml --epochs 100 --batch 16 --imgsz 640
```

**Các options:**
- `--data`: Path đến data.yaml
- `--epochs`: Số epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--device`: Device (None để auto-detect GPU/CPU, '0' cho GPU, 'cpu' cho CPU)

### 4. Inference và tạo predictions

Generate predictions cho test set:

```bash
python test.py --model runs/detect/object_detection/weights/best.pt --samples public_test/public_test/samples --output predictions.json
```

**Các options:**
- `--model`: Path đến trained model
- `--samples`: Path đến test samples directory
- `--output`: Output file cho predictions
- `--conf`: Confidence threshold (default: 0.5)

## Quy trình đầy đủ

### Cách 1: Chạy từng bước

```bash
# Bước 1: Cài đặt
pip install -r requirements.txt

# Bước 2: Chuẩn bị dataset (chỉ cần chạy 1 lần)
python prepare_dataset.py

# Bước 3: Train model
python train.py --epochs 100 --batch 16

# Bước 4: Inference và tạo predictions
python test.py --model runs/detect/object_detection/weights/best.pt --samples public_test/public_test/samples --output predictions.json
```

### Cách 2: Chạy tất cả với một lệnh

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy toàn bộ pipeline
python run_all.py
```

### Scripts bổ trợ

```bash
# Phân tích dataset
python utils.py --annotations observing/train/annotations/annotations.json --samples observing/train/samples

# Visualize annotations trên video
python visualize_annotations.py --video-id Backpack_0
```

## Format dữ liệu

### Annotations (ground truth)
```json
{
  "video_id": "Backpack_0",
  "annotations": [{
    "bboxes": [
      {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
      {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
    ]
  }]
}
```

### Predictions (output)
```json
[
  {
    "video_id": "BlackBox_0",
    "detections": [{
      "bboxes": [
        {"frame": 100, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
      ]
    }]
  },
  {
    "video_id": "BlackBox_1",
    "detections": []
  }
]
```

## Chiến lược "YOLO-như-Đối sánh-Mẫu"

**Ý tưởng:** Sử dụng 3 ảnh tham chiếu để tạo dataset tổng hợp lớn, sau đó train một mô hình YOLOv8n (nano - nhẹ nhất) để hoạt động như một bộ phát hiện chuyên biệt cho vật thể mục tiêu.

**Ưu điểm:**
- Tốc độ cao: YOLOv8n có thể chạy >15 FPS trên Jetson
- Chuyên biệt: Model chỉ detect vật thể mục tiêu
- Nhẹ: Model nhỏ gọn, phù hợp cho edge devices
- Dễ triển khai: Không cần template matching phức tạp

## Model Architecture

- **Base Model**: YOLOv8n (Ultralytics)
- **Classes**: 1 (target_object)
- **Image Size**: 640x640
- **Output**: Bounding boxes với confidence scores

## Kết quả

Model được lưu tại: `runs/detect/object_detection/weights/best.pt`

Các metrics training:
- Precision, Recall, mAP50, mAP50-95
- Loss curves
- Được lưu tự động trong quá trình training

## Troubleshooting

### GPU không được detect
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Train trên CPU
python train.py --device cpu
```

**Lưu ý:** Mặc định code sẽ tự động detect GPU/CPU. Nếu không có GPU, sẽ tự động dùng CPU.

### Out of memory
- Giảm batch size: `--batch 8`
- Giảm image size: `--imgsz 416`

### Dataset quá lớn
- Dùng option `--sampling-rate 20` khi prepare dataset để lấy ít frames hơn

## Lưu ý

- Video trong dataset thường có 3-5 phút (25 FPS)
- Mỗi sample có 3 ảnh tham chiếu từ ground-level viewpoints
- Model được train theo single-class detection
- Tất cả video phải có trong predictions output, kể cả không detect được object
- Code tự động detect GPU/CPU, không cần chỉ định device

## Tác giả

Approach này được thiết kế để tối ưu cho việc triển khai trên NVIDIA Jetson với yêu cầu performance >15 FPS.
