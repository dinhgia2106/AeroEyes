"""
Script test trên bộ data public_test theo pipeline:

Bước 1: Khởi tạo (Mỗi video một lần):
- Load mô hình Siamese-YOLOv8 đã được huấn luyện
- Đưa 3 ảnh tham chiếu qua backbone để tạo Query Vector Q_v
- Khởi tạo SORT Tracker

Bước 2: Thực thi (Trên từng khung hình):
- Giải mã video frame by frame
- Với mỗi frame, đưa frame và Q_v vào mô hình
- Mô hình trả về raw detections
- Raw detections được đưa vào SORT Tracker

Bước 3: Tổng hợp Kết quả:
- Thu thập đầu ra từ SORT tracker
- Format theo JSON schema
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse

from models.siamese_yolo import SiameseYOLOv8
from tracking.sort_tracker import SORTTracker


class PostProcessor:
    """Post-process model outputs để lấy bounding boxes."""
    
    def __init__(
        self,
        img_size: int = 640,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        similarity_threshold: float = 0.5
    ):
        """
        Args:
            img_size: Kích thước input image
            conf_threshold: Ngưỡng confidence (objectness * similarity)
            iou_threshold: Ngưỡng IoU cho NMS
            similarity_threshold: Ngưỡng similarity score
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold
        
        # Anchor scales cho 3 levels (P3, P4, P5)
        # YOLOv8 sử dụng anchor-free, nhưng chúng ta cần decode từ grid cells
        self.strides = [8, 16, 32]  # Stride cho P3, P4, P5
    
    def decode_bbox(
        self,
        bbox_pred: torch.Tensor,
        obj_pred: torch.Tensor,
        sim_pred: torch.Tensor,
        scale_idx: int
    ) -> np.ndarray:
        """
        Decode bounding boxes từ predictions.
        
        Args:
            bbox_pred: (1, 4, H, W) - Bbox predictions [cx, cy, w, h] (normalized)
            obj_pred: (1, 1, H, W) - Objectness predictions
            sim_pred: (1, 1, H, W) - Similarity predictions
            scale_idx: Index của scale (0=P3, 1=P4, 2=P5)
        
        Returns:
            detections: (N, 5) - [x1, y1, x2, y2, conf]
        """
        stride = self.strides[scale_idx]
        B, C, H, W = bbox_pred.shape
        
        # Get predictions
        bbox_pred = bbox_pred[0].cpu().numpy()  # (4, H, W)
        obj_pred = obj_pred[0, 0].cpu().numpy()  # (H, W)
        sim_pred = sim_pred[0, 0].cpu().numpy()  # (H, W)
        
        # Apply sigmoid to objectness và similarity (nếu chưa được apply)
        obj_pred = 1 / (1 + np.exp(-np.clip(obj_pred, -10, 10)))  # Sigmoid
        sim_pred = np.clip(sim_pred, -1, 1)  # Similarity thường trong [-1, 1]
        sim_pred = (sim_pred + 1) / 2  # Normalize to [0, 1]
        
        # Combined confidence
        conf = obj_pred * sim_pred
        
        # Find cells với confidence > threshold
        mask = conf > self.conf_threshold
        
        if not np.any(mask):
            return np.empty((0, 5), dtype=np.float32)
        
        # Get grid coordinates
        grid_y, grid_x = np.where(mask)
        
        # Decode bboxes
        # Bbox predictions có thể là [cx, cy, w, h] hoặc [x1, y1, x2, y2]
        # Giả sử format là [cx, cy, w, h] normalized
        cx_offset = bbox_pred[0, grid_y, grid_x]  # Offset từ grid center
        cy_offset = bbox_pred[1, grid_y, grid_x]
        w_norm = bbox_pred[2, grid_y, grid_x]   # Width (normalized)
        h_norm = bbox_pred[3, grid_y, grid_x]   # Height (normalized)
        
        # Apply sigmoid cho offsets (nếu cần)
        cx_offset = np.clip(cx_offset, -0.5, 0.5)  # Offset thường trong [-0.5, 0.5]
        cy_offset = np.clip(cy_offset, -0.5, 0.5)
        
        # Convert to absolute coordinates
        cx = (grid_x + 0.5 + cx_offset) * stride
        cy = (grid_y + 0.5 + cy_offset) * stride
        w = w_norm * self.img_size
        h = h_norm * self.img_size
        
        # Convert to [x1, y1, x2, y2]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, self.img_size - 1)
        y1 = np.clip(y1, 0, self.img_size - 1)
        x2 = np.clip(x2, 0, self.img_size - 1)
        y2 = np.clip(y2, 0, self.img_size - 1)
        
        # Get confidences
        confidences = conf[grid_y, grid_x]
        
        # Stack: [x1, y1, x2, y2, conf]
        detections = np.stack([x1, y1, x2, y2, confidences], axis=1)
        
        return detections.astype(np.float32)
    
    def nms(self, detections: np.ndarray) -> np.ndarray:
        """
        Non-Maximum Suppression.
        
        Args:
            detections: (N, 5) - [x1, y1, x2, y2, conf]
        
        Returns:
            keep_indices: Indices của detections được giữ lại
        """
        if len(detections) == 0:
            return np.array([], dtype=np.int32)
        
        # Sort by confidence
        order = np.argsort(detections[:, 4])[::-1]
        keep = []
        
        while len(order) > 0:
            # Lấy detection có confidence cao nhất
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Tính IoU với các detections còn lại
            ious = self._compute_iou(detections[i], detections[order[1:]])
            
            # Giữ lại những detections có IoU < threshold
            mask = ious < self.iou_threshold
            order = order[1:][mask]
        
        return np.array(keep, dtype=np.int32)
    
    def _compute_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Tính IoU giữa box1 và boxes."""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def process(
        self,
        obj_preds: List[torch.Tensor],
        sim_preds: List[torch.Tensor],
        bbox_preds: List[torch.Tensor]
    ) -> List[np.ndarray]:
        """
        Process model outputs để lấy detections.
        
        Args:
            obj_preds: List of (1, 1, H, W) - Objectness predictions cho 3 scales
            sim_preds: List of (1, 1, H, W) - Similarity predictions cho 3 scales
            bbox_preds: List of (1, 4, H, W) - Bbox predictions cho 3 scales
        
        Returns:
            detections: List of (N, 4) - [x1, y1, x2, y2] (sau NMS)
        """
        all_detections = []
        
        # Decode từ mỗi scale
        for scale_idx in range(len(obj_preds)):
            detections = self.decode_bbox(
                bbox_preds[scale_idx],
                obj_preds[scale_idx],
                sim_preds[scale_idx],
                scale_idx
            )
            if len(detections) > 0:
                all_detections.append(detections)
        
        if len(all_detections) == 0:
            return []
        
        # Concatenate tất cả detections
        all_detections = np.concatenate(all_detections, axis=0)
        
        # Apply NMS
        keep_indices = self.nms(all_detections)
        
        if len(keep_indices) == 0:
            return []
        
        # Return bboxes (không có confidence)
        final_detections = all_detections[keep_indices, :4]  # (N, 4)
        
        return [final_detections]


def preprocess_image(img: np.ndarray, img_size: int = 640) -> torch.Tensor:
    """
    Preprocess image cho model.
    
    Args:
        img: (H, W, 3) - BGR image
        img_size: Target size
    
    Returns:
        tensor: (1, 3, img_size, img_size) - Normalized tensor
    """
    # Resize
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # To tensor
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor


def test_public_test(
    model_path: str,
    public_test_dir: str,
    output_path: str,
    device: str = 'cuda',
    img_size: int = 640,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45
):
    """
    Test trên bộ data public_test.
    
    Args:
        model_path: Đường dẫn đến model checkpoint
        public_test_dir: Đường dẫn đến thư mục public_test
        output_path: Đường dẫn output JSON file
        device: 'cuda' hoặc 'cpu'
        img_size: Kích thước input image
        conf_threshold: Ngưỡng confidence
        iou_threshold: Ngưỡng IoU cho NMS
    """
    print("🚀 Bắt đầu test trên public_test...")
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from {model_path}...")
    model = SiameseYOLOv8(model_size='n', feature_dim=256, num_ref_images=3)
    model.to(device)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Khởi tạo heads bằng cách forward một dummy input
        print("   🔧 Initializing model heads...")
        dummy_video = torch.randn(1, 3, img_size, img_size).to(device)
        dummy_refs = torch.randn(1, 3, 3, img_size, img_size).to(device)
        with torch.no_grad():
            _ = model(dummy_video, dummy_refs)
        
        # Load state dict với strict=False để bỏ qua các keys không khớp
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"   ⚠️  Missing keys (sẽ sử dụng giá trị mặc định): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys (sẽ bỏ qua): {len(unexpected_keys)} keys")
    else:
        print(f"   ⚠️  Warning: Model file không tồn tại, sử dụng model chưa train")
    
    model.eval()
    print("✅ Model loaded!")
    
    # Post-processor
    post_processor = PostProcessor(
        img_size=img_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # Tìm tất cả videos trong public_test
    public_test_path = Path(public_test_dir)
    samples_dir = public_test_path / "public_test" / "samples"
    
    if not samples_dir.exists():
        samples_dir = public_test_path / "samples"
    
    video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    print(f"📹 Tìm thấy {len(video_dirs)} videos")
    
    results = []
    
    # Xử lý từng video
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir.name
        print(f"\n📹 Processing video: {video_id}")
        
        # Bước 1: Khởi tạo
        # Load 3 reference images
        ref_images = []
        for i in range(1, 4):
            ref_path = video_dir / "object_images" / f"img_{i}.jpg"
            if not ref_path.exists():
                print(f"⚠️  Warning: {ref_path} không tồn tại, bỏ qua video {video_id}")
                results.append({
                    "video_id": video_id,
                    "detections": []
                })
                break
            
            ref_img = cv2.imread(str(ref_path))
            if ref_img is None:
                print(f"⚠️  Warning: Không thể đọc {ref_path}, bỏ qua video {video_id}")
                results.append({
                    "video_id": video_id,
                    "detections": []
                })
                break
            
            ref_images.append(ref_img)
        
        if len(ref_images) != 3:
            continue
        
        # Preprocess reference images
        ref_tensors = [preprocess_image(img, img_size) for img in ref_images]
        ref_tensors = torch.stack(ref_tensors, dim=1)  # (1, 3, 3, H, W)
        ref_tensors = ref_tensors.to(device)
        
        # Extract query vector (một lần cho mỗi video)
        with torch.no_grad():
            query_vector = model.extract_reference_features(ref_tensors)  # (1, D)
        print(f"✅ Query vector extracted: {query_vector.shape}")
        
        # Khởi tạo SORT Tracker
        tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Bước 2: Thực thi trên từng frame
        video_path = video_dir / "drone_video.mp4"
        if not video_path.exists():
            print(f"⚠️  Warning: {video_path} không tồn tại, bỏ qua video {video_id}")
            results.append({
                "video_id": video_id,
                "detections": []
            })
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️  Warning: Không thể mở {video_path}, bỏ qua video {video_id}")
            results.append({
                "video_id": video_id,
                "detections": []
            })
            continue
        
        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Video info: {width}x{height}, {total_frames} frames, {fps:.2f} fps")
        
        # Lưu trữ detections cho tất cả frames theo track_id
        all_track_detections = {}  # {track_id: {frame_id: bbox}}
        
        frame_id = 0
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor = preprocess_image(frame, img_size).to(device)
                
                # Forward pass
                # Model cần cả video_frame và reference_images
                obj_preds, sim_preds, bbox_preds = model(frame_tensor, ref_tensors)
                
                # Post-process
                detections_list = post_processor.process(obj_preds, sim_preds, bbox_preds)
                
                # Convert detections sang format cho SORT tracker
                if len(detections_list) > 0 and len(detections_list[0]) > 0:
                    detections = detections_list[0]  # (N, 4) - [x1, y1, x2, y2]
                    # Scale về kích thước gốc của video
                    scale_x = width / img_size
                    scale_y = height / img_size
                    detections_scaled = detections.copy()
                    detections_scaled[:, [0, 2]] *= scale_x
                    detections_scaled[:, [1, 3]] *= scale_y
                    detections_scaled = detections_scaled.astype(np.float32)
                    detections_for_tracker = [det for det in detections_scaled]
                else:
                    detections_for_tracker = []
                
                # Update SORT tracker
                tracks = tracker.update(detections_for_tracker)
                
                # Lưu detections từ tracker theo track_id
                for track in tracks:
                    track_id = track['track_id']
                    bbox = track['bbox']  # [x1, y1, x2, y2]
                    
                    # Đảm bảo bbox trong phạm vi
                    x1 = max(0, min(width - 1, int(round(bbox[0]))))
                    y1 = max(0, min(height - 1, int(round(bbox[1]))))
                    x2 = max(0, min(width - 1, int(round(bbox[2]))))
                    y2 = max(0, min(height - 1, int(round(bbox[3]))))
                    
                    if track_id not in all_track_detections:
                        all_track_detections[track_id] = {}
                    
                    all_track_detections[track_id][frame_id] = {
                        "frame": frame_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                
                frame_id += 1
        
        cap.release()
        
        # Bước 3: Tổng hợp kết quả
        # Format theo schema yêu cầu
        if len(all_track_detections) > 0:
            # Mỗi track là một detection
            detections_list = []
            for track_id in sorted(all_track_detections.keys()):
                track_bboxes = all_track_detections[track_id]
                # Sắp xếp bboxes theo frame_id
                bboxes_list = [
                    track_bboxes[frame_id]
                    for frame_id in sorted(track_bboxes.keys())
                ]
                detections_list.append({
                    "bboxes": bboxes_list
                })
            
            results.append({
                "video_id": video_id,
                "detections": detections_list
            })
        else:
            results.append({
                "video_id": video_id,
                "detections": []
            })
        
        num_frames_with_detections = sum(
            len(frames) for frames in all_track_detections.values()
        )
        print(f"   ✅ Processed {frame_id} frames, {len(all_track_detections)} tracks, {num_frames_with_detections} frame detections")
    
    # Lưu kết quả
    print(f"\n💾 Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Hoàn thành! Đã xử lý {len(results)} videos")
    print(f"   Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trên public_test dataset")
    parser.add_argument(
        "--model",
        default="runs/siamese_tracking/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--public-test-dir",
        default="public_test",
        help="Path to public_test directory"
    )
    parser.add_argument(
        "--output",
        default="predictions.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    
    args = parser.parse_args()
    
    test_public_test(
        model_path=args.model,
        public_test_dir=args.public_test_dir,
        output_path=args.output,
        device=args.device,
        img_size=args.img_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )

