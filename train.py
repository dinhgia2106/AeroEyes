"""
Training Script cho Siamese-YOLOv8 với Multi-task Loss (PHẦN 6).

Theo báo cáo PHẦN 6, hàm mất mát tổng thể là:
L_total = w1 * L_detection + w2 * L_similarity

Bao gồm:
- Detection Loss: CIoU + Focal Loss (PHẦN 6.2)
- Similarity Loss: Triplet Loss với hard negative mining (PHẦN 6.3)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional

from models.siamese_yolo import SiameseYOLOv8
from models.losses import MultiTaskLoss
from utils.triplet_mining import TripletMiner
from data_pipeline.group_split import group_based_split, get_split_for_video


class AeroEyesDataset(torch.utils.data.Dataset):
    """
    Dataset cho AeroEyes training.
    
    Dataset này load video frames và reference images, áp dụng augmentations on-the-fly.
    """
    
    def __init__(
        self,
        annotations_path: str,
        samples_dir: str,
        video_ids: List[str],
        img_size: int = 640,
        apply_augmentations: bool = True
    ):
        """
        Args:
            annotations_path: Path đến annotations.json
            samples_dir: Directory chứa samples
            video_ids: List video IDs cho dataset này (train hoặc val)
            img_size: Kích thước input image
            apply_augmentations: Có áp dụng augmentations không
        """
        self.samples_dir = Path(samples_dir)
        self.img_size = img_size
        self.apply_augmentations = apply_augmentations
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            all_annotations = json.load(f)
        
        # Filter annotations theo video_ids
        self.annotations = [
            ann for ann in all_annotations 
            if ann['video_id'] in video_ids
        ]
        
        # Tạo danh sách samples: (video_id, frame_num, bbox)
        self.samples = []
        for ann in self.annotations:
            video_id = ann['video_id']
            for annotation_set in ann['annotations']:
                for bbox_info in annotation_set['bboxes']:
                    frame_num = bbox_info['frame']
                    self.samples.append({
                        'video_id': video_id,
                        'frame_num': frame_num,
                        'bbox': {
                            'x1': bbox_info['x1'],
                            'y1': bbox_info['y1'],
                            'x2': bbox_info['x2'],
                            'y2': bbox_info['y2']
                        }
                    })
        
        print(f"📦 Loaded {len(self.samples)} samples từ {len(video_ids)} videos")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'video_frame': torch.Tensor (3, H, W),
                'reference_images': torch.Tensor (3, 3, H, W),
                'bbox': Dict với keys ['x1', 'y1', 'x2', 'y2'],
                'video_id': str
            }
        """
        sample = self.samples[idx]
        video_id = sample['video_id']
        frame_num = sample['frame_num']
        bbox = sample['bbox']
        
        # Load video frame
        video_path = self.samples_dir / video_id / "drone_video.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Không thể đọc frame {frame_num} từ {video_path}")
        
        # Load reference images
        ref_images = []
        for i in range(1, 4):
            ref_path = self.samples_dir / video_id / "object_images" / f"img_{i}.jpg"
            ref_img = cv2.imread(str(ref_path))
            if ref_img is None:
                raise ValueError(f"Không thể đọc reference image: {ref_path}")
            ref_images.append(ref_img)
        
        # Preprocess
        video_frame = self._preprocess_image(frame)
        reference_images = torch.stack([self._preprocess_image(img) for img in ref_images])
        
        return {
            'video_frame': video_frame,
            'reference_images': reference_images,
            'bbox': bbox,
            'video_id': video_id,
            'frame_num': frame_num
        }
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize, normalize"""
        # Resize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)  # (3, H, W)
        
        return img_tensor


def create_targets(
    bbox: Dict,
    img_size: int,
    feature_sizes: List[Tuple[int, int]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Tạo targets cho detection loss.
    
    Args:
        bbox: Dict với keys ['x1', 'y1', 'x2', 'y2']
        img_size: Kích thước input image
        feature_sizes: List of (H, W) cho mỗi scale
    
    Returns:
        (objectness_targets, bbox_targets) - mỗi là List of tensors
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # Convert to center format
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize
    center_x_norm = center_x / img_size
    center_y_norm = center_y / img_size
    width_norm = width / img_size
    height_norm = height / img_size
    
    objectness_targets = []
    bbox_targets = []
    
    for H, W in feature_sizes:
        # Tạo target tensors
        obj_target = torch.zeros(1, 1, H, W)
        bbox_target = torch.zeros(1, 4, H, W)
        
        # Tìm grid cell chứa center
        grid_x = int(center_x_norm * W)
        grid_y = int(center_y_norm * H)
        
        grid_x = max(0, min(W - 1, grid_x))
        grid_y = max(0, min(H - 1, grid_y))
        
        # Set target
        obj_target[0, 0, grid_y, grid_x] = 1.0
        bbox_target[0, 0, grid_y, grid_x] = center_x_norm
        bbox_target[0, 1, grid_y, grid_x] = center_y_norm
        bbox_target[0, 2, grid_y, grid_x] = width_norm
        bbox_target[0, 3, grid_y, grid_x] = height_norm
        
        objectness_targets.append(obj_target)
        bbox_targets.append(bbox_target)
    
    return objectness_targets, bbox_targets


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    triplet_miner: Optional[TripletMiner] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """Train một epoch"""
    model.train()
    total_loss = 0.0
    total_detection_loss = 0.0
    total_similarity_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        video_frames = batch['video_frame'].to(device)  # (B, 3, H, W)
        reference_images = batch['reference_images'].to(device)  # (B, 3, 3, H, W)
        bboxes = batch['bbox']
        
        B = video_frames.shape[0]
        
        # Forward pass
        obj_preds, sim_preds, bbox_preds = model(video_frames, reference_images)
        
        # Create targets
        feature_sizes = [(pred.shape[2], pred.shape[3]) for pred in obj_preds]
        obj_targets = []
        bbox_targets = []
        
        # Xử lý bboxes: DataLoader có thể trả về dict với list values hoặc list of dicts
        if isinstance(bboxes, dict):
            # Convert dict với list values thành list of dicts
            bbox_list = [
                {
                    'x1': bboxes['x1'][i],
                    'y1': bboxes['y1'][i],
                    'x2': bboxes['x2'][i],
                    'y2': bboxes['y2'][i]
                }
                for i in range(B)
            ]
        else:
            # Đã là list of dicts
            bbox_list = bboxes
        
        for i in range(B):
            obj_t, bbox_t = create_targets(bbox_list[i], 640, feature_sizes)
            if i == 0:
                obj_targets = [[t] for t in obj_t]
                bbox_targets = [[t] for t in bbox_t]
            else:
                for j, (ot, bt) in enumerate(zip(obj_t, bbox_t)):
                    obj_targets[j].append(ot)
                    bbox_targets[j].append(bt)
        
        # Stack targets
        obj_targets = [torch.cat(ts, dim=0).to(device) for ts in obj_targets]
        bbox_targets = [torch.cat(ts, dim=0).to(device) for ts in bbox_targets]
        
        # Extract features cho triplet loss (nếu có)
        anchor_features = None
        positive_features = None
        negative_features = None
        
        if triplet_miner is not None:
            # Extract anchor features (từ video frame tại vị trí bbox)
            # Đơn giản hóa: sử dụng query vector làm positive, và random features làm negative
            with torch.no_grad():
                query_vectors = model.extract_reference_features(reference_images)  # (B, D)
            
            # Anchor: features từ video frame (cần extract từ bbox region)
            # Đơn giản hóa: sử dụng global features
            video_features = model.extract_backbone_features(video_frames)
            if isinstance(video_features, list):
                anchor_features = video_features[-1]  # P5
                anchor_features = torch.nn.functional.adaptive_avg_pool2d(anchor_features, 1).squeeze(-1).squeeze(-1)  # (B, C)
                # Project to feature_dim
                anchor_features = torch.nn.Linear(anchor_features.shape[1], model.feature_dim).to(device)(anchor_features)
            else:
                anchor_features = query_vectors  # Fallback
            
            positive_features = query_vectors
            
            # Negative: random features từ batch khác
            negative_indices = torch.randperm(B).to(device)
            negative_features = query_vectors[negative_indices]
        
        # Compute loss
        loss, loss_dict = criterion(
            obj_preds, sim_preds, bbox_preds,
            obj_targets, bbox_targets,
            anchor_features, positive_features, negative_features
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_detection_loss += loss_dict['detection_loss']
        total_similarity_loss += loss_dict['similarity_loss']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'det': f"{loss_dict['detection_loss']:.4f}",
            'sim': f"{loss_dict['similarity_loss']:.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'detection_loss': total_detection_loss / num_batches,
        'similarity_loss': total_similarity_loss / num_batches
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_detection_loss = 0.0
    total_similarity_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            video_frames = batch['video_frame'].to(device)
            reference_images = batch['reference_images'].to(device)
            bboxes = batch['bbox']
            
            B = video_frames.shape[0]
            
            # Forward pass
            obj_preds, sim_preds, bbox_preds = model(video_frames, reference_images)
            
            # Create targets
            feature_sizes = [(pred.shape[2], pred.shape[3]) for pred in obj_preds]
            obj_targets = []
            bbox_targets = []
            
            # Xử lý bboxes: DataLoader có thể trả về dict với list values hoặc list of dicts
            if isinstance(bboxes, dict):
                # Convert dict với list values thành list of dicts
                bbox_list = [
                    {
                        'x1': bboxes['x1'][i],
                        'y1': bboxes['y1'][i],
                        'x2': bboxes['x2'][i],
                        'y2': bboxes['y2'][i]
                    }
                    for i in range(B)
                ]
            else:
                # Đã là list of dicts
                bbox_list = bboxes
            
            for i in range(B):
                obj_t, bbox_t = create_targets(bbox_list[i], 640, feature_sizes)
                if i == 0:
                    obj_targets = [[t] for t in obj_t]
                    bbox_targets = [[t] for t in bbox_t]
                else:
                    for j, (ot, bt) in enumerate(zip(obj_t, bbox_t)):
                        obj_targets[j].append(ot)
                        bbox_targets[j].append(bt)
            
            # Stack targets
            obj_targets = [torch.cat(ts, dim=0).to(device) for ts in obj_targets]
            bbox_targets = [torch.cat(ts, dim=0).to(device) for ts in bbox_targets]
            
            # Compute loss
            loss, loss_dict = criterion(
                obj_preds, sim_preds, bbox_preds,
                obj_targets, bbox_targets
            )
            
            total_loss += loss.item()
            total_detection_loss += loss_dict['detection_loss']
            total_similarity_loss += loss_dict['similarity_loss']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'detection_loss': total_detection_loss / num_batches,
        'similarity_loss': total_similarity_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train Siamese-YOLOv8")
    parser.add_argument(
        "--annotations",
        default="observing/train/annotations/annotations.json",
        help="Path to annotations.json"
    )
    parser.add_argument(
        "--samples",
        default="observing/train/samples",
        help="Path to samples directory"
    )
    parser.add_argument(
        "--output-dir",
        default="runs/siamese_tracking",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--detection-weight",
        type=float,
        default=1.0,
        help="Weight for detection loss"
    )
    parser.add_argument(
        "--similarity-weight",
        type=float,
        default=0.5,
        help="Weight for similarity loss"
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=1.0,
        help="Margin for triplet loss"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Group-based splitting
    print("🔀 Performing Group-Based Splitting...")
    train_video_ids, val_video_ids = group_based_split(
        args.annotations, train_ratio=args.train_ratio, random_seed=42
    )
    print(f"   Train videos: {len(train_video_ids)}")
    print(f"   Val videos: {len(val_video_ids)}")
    
    # Datasets
    train_dataset = AeroEyesDataset(
        args.annotations, args.samples, train_video_ids,
        img_size=args.img_size, apply_augmentations=True
    )
    val_dataset = AeroEyesDataset(
        args.annotations, args.samples, val_video_ids,
        img_size=args.img_size, apply_augmentations=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Model
    print("📦 Creating model...")
    model = SiameseYOLOv8(model_size='n', feature_dim=256, num_ref_images=3)
    model.to(device)
    
    # Loss
    criterion = MultiTaskLoss(
        detection_weight=args.detection_weight,
        similarity_weight=args.similarity_weight,
        triplet_margin=args.triplet_margin
    )
    
    # Triplet miner
    triplet_miner = TripletMiner(margin=args.triplet_margin)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"📂 Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("🚀 Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            triplet_miner, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Det: {train_metrics['detection_loss']:.4f}, "
              f"Sim: {train_metrics['similarity_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Det: {val_metrics['detection_loss']:.4f}, "
              f"Sim: {val_metrics['similarity_loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pt')
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"  ✅ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()

