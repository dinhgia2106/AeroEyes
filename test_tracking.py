"""
Script để inference và tạo predictions từ trained tracking model
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
from model import TrackingModel
import math


class VideoTracker:
    """Video tracker sử dụng mô hình tracking"""
    def __init__(self, model_path, device=None, conf_threshold=0.5):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.img_size = 256
        
        # Load model
        self.model = TrackingModel(num_classes=1, num_anchors=9)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {device}")
    
    def load_frame(self, cap, frame_num):
        """Load frame từ video"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(self, img, bbox=None, context=0.5):
        """Preprocess image"""
        h, w = img.shape[:2]
        
        if bbox is not None:
            # Crop với context
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            context_w = int(bbox_w * context)
            context_h = int(bbox_h * context)
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            crop_x1 = max(0, cx - bbox_w // 2 - context_w)
            crop_y1 = max(0, cy - bbox_h // 2 - context_h)
            crop_x2 = min(w, cx + bbox_w // 2 + context_w)
            crop_y2 = min(h, cy + bbox_h // 2 + context_h)
            
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
        else:
            # Use full image
            crop = img
            crop_coords = (0, 0, w, h)
        
        # Resize
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        
        # To tensor and normalize
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        
        return tensor, crop_coords
    
    def decode_bbox(self, reg_pred, cls_pred, anchors, crop_coords, original_size):
        """Decode bbox từ predictions"""
        B, num_anchors, H, W = cls_pred.shape
        
        # Get best anchor
        cls_scores = torch.sigmoid(cls_pred)  # [B, num_anchors, H, W]
        max_scores, best_anchors = torch.max(cls_scores.view(B, -1), dim=1)
        
        if max_scores[0] < self.conf_threshold:
            return None
        
        # Get best anchor index
        best_anchor_idx = best_anchors[0].item() % num_anchors
        best_pos = best_anchors[0].item()
        
        # Get position in feature map
        pos_h = best_pos // (num_anchors * W)
        pos_w = (best_pos % (num_anchors * W)) // num_anchors
        
        # Get regression
        reg = reg_pred[0, best_anchor_idx*4:(best_anchor_idx+1)*4, pos_h, pos_w]
        cx, cy, w, h = reg.cpu().numpy()
        
        # Convert to absolute coordinates
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        # Scale to crop size
        cx_abs = cx * crop_w + crop_x1
        cy_abs = cy * crop_h + crop_y1
        w_abs = w * crop_w
        h_abs = h * crop_h
        
        # Convert to x1, y1, x2, y2
        x1 = int(cx_abs - w_abs / 2)
        y1 = int(cy_abs - h_abs / 2)
        x2 = int(cx_abs + w_abs / 2)
        y2 = int(cy_abs + h_abs / 2)
        
        # Clip to image bounds
        orig_h, orig_w = original_size
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'score': float(max_scores[0].item())
        }
    
    def track_video(self, video_path):
        """Track object trong video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Load first frame as template
        template_frame = self.load_frame(cap, 0)
        if template_frame is None:
            cap.release()
            return []
        
        # Initialize with first detection (if available) or use center crop
        # For now, use center crop as template
        h, w = template_frame.shape[:2]
        initial_bbox = {
            'x1': w // 4,
            'y1': h // 4,
            'x2': 3 * w // 4,
            'y2': 3 * h // 4
        }
        
        template_tensor, _ = self.preprocess_image(template_frame, initial_bbox)
        template_tensor = template_tensor.unsqueeze(0).to(self.device)
        
        # Track through video
        detections = []
        current_bbox = initial_bbox
        
        self.model.reset_hidden_state()
        
        with torch.no_grad():
            for frame_num in tqdm(range(total_frames), desc=f"Tracking {video_path.name}"):
                search_frame = self.load_frame(cap, frame_num)
                if search_frame is None:
                    break
                
                # Preprocess search frame
                search_tensor, crop_coords = self.preprocess_image(search_frame, current_bbox)
                search_tensor = search_tensor.unsqueeze(0).to(self.device)
                
                # Forward pass
                cls_pred, reg_pred = self.model(template_tensor, search_tensor, reset_hidden=False)
                
                # Generate anchors
                anchors = self.model.generate_anchors(
                    feature_size=(cls_pred.shape[2], cls_pred.shape[3]),
                    stride=16
                ).to(self.device)
                
                # Decode bbox
                bbox = self.decode_bbox(
                    reg_pred, cls_pred, anchors, crop_coords,
                    (search_frame.shape[0], search_frame.shape[1])
                )
                
                if bbox is not None:
                    detection = {
                        'frame': frame_num,
                        'x1': bbox['x1'],
                        'y1': bbox['y1'],
                        'x2': bbox['x2'],
                        'y2': bbox['y2']
                    }
                    detections.append(detection)
                    
                    # Update current bbox for next frame
                    current_bbox = {
                        'x1': bbox['x1'],
                        'y1': bbox['y1'],
                        'x2': bbox['x2'],
                        'y2': bbox['y2']
                    }
        
        cap.release()
        
        # Group consecutive detections
        grouped = self.group_consecutive_detections(detections)
        
        return grouped
    
    def group_consecutive_detections(self, detections):
        """Group consecutive detections"""
        if not detections:
            return []
        
        grouped = []
        current_group = []
        
        for det in detections:
            if not current_group:
                current_group.append(det)
            else:
                last_frame = current_group[-1]['frame']
                if det['frame'] - last_frame <= 10:
                    current_group.append(det)
                else:
                    if current_group:
                        grouped.append({"bboxes": current_group})
                    current_group = [det]
        
        if current_group:
            grouped.append({"bboxes": current_group})
        
        return grouped


def generate_predictions(model_path, test_samples_dir, output_file="predictions.json"):
    """Generate predictions cho tất cả videos"""
    tracker = VideoTracker(model_path)
    
    samples_dir = Path(test_samples_dir)
    if not samples_dir.exists():
        print(f"Error: {samples_dir} not found!")
        return
    
    video_samples = sorted(list(samples_dir.glob("*")))
    print(f"Found {len(video_samples)} video samples")
    
    all_predictions = []
    
    for sample_dir in tqdm(video_samples, desc="Processing test samples"):
        video_path = sample_dir / "drone_video.mp4"
        
        if not video_path.exists():
            print(f"Warning: {video_path} not found")
            continue
        
        video_id = sample_dir.name
        
        # Track video
        detections = tracker.track_video(video_path)
        
        prediction = {
            "video_id": video_id,
            "detections": detections if detections else []
        }
        
        all_predictions.append(prediction)
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nPredictions saved to: {output_file}")
    
    # Summary
    total_with_detections = sum(1 for pred in all_predictions if pred['detections'])
    print(f"Summary:")
    print(f"  Total videos: {len(all_predictions)}")
    print(f"  Videos with detections: {total_with_detections}")
    print(f"  Videos without detections: {len(all_predictions) - total_with_detections}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions from trained tracking model")
    parser.add_argument("--model", default="runs/tracking/best.pt",
                       help="Path to trained model")
    parser.add_argument("--samples", default="public_test/public_test/samples",
                       help="Path to test samples directory")
    parser.add_argument("--output", default="predictions.json",
                       help="Output file for predictions")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--device", default=None,
                       help="Device to use ('cuda' or 'cpu', None for auto-detect)")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} not found!")
        print("Please train the model first using train_tracking.py")
        exit(1)
    
    generate_predictions(
        model_path=str(model_path),
        test_samples_dir=args.samples,
        output_file=args.output
    )

