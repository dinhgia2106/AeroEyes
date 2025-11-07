"""
Script để train Tracking Model với:
- Backbone: CSPNet-Tiny
- Neck: BiFPN-Lite
- Temporal: ConvGRU
- Head: Depthwise Cross-Correlation + RPN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import json
import numpy as np
from tqdm import tqdm
import random
from model import TrackingModel, count_parameters


class TrackingDataset(Dataset):
    """Dataset cho object tracking"""
    def __init__(self, annotations_path, samples_dir, img_size=256, augment=True):
        self.img_size = img_size
        self.augment = augment
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.samples_dir = Path(samples_dir)
        
        # Prepare samples
        self.samples = []
        for ann in self.annotations:
            video_id = ann['video_id']
            video_path = self.samples_dir / video_id / "drone_video.mp4"
            
            if not video_path.exists():
                continue
            
            # Process each detection interval
            for det_interval in ann['annotations']:
                bboxes = det_interval['bboxes']
                if len(bboxes) < 2:  # Need at least 2 frames
                    continue
                
                self.samples.append({
                    'video_path': video_path,
                    'bboxes': bboxes,
                    'video_id': video_id
                })
    
    def __len__(self):
        return len(self.samples)
    
    def load_frame(self, video_path, frame_num):
        """Load frame từ video"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def crop_bbox(self, img, bbox, context=0.5):
        """Crop bbox với context"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Add context
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
        
        # Resize to img_size
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        
        return crop, (crop_x1, crop_y1, crop_x2, crop_y2)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        bboxes = sample['bboxes']
        
        # Select template and search frames
        if len(bboxes) == 2:
            template_idx, search_idx = 0, 1
        else:
            template_idx = random.randint(0, len(bboxes) - 2)
            search_idx = random.randint(template_idx + 1, len(bboxes) - 1)
        
        template_bbox = bboxes[template_idx]
        search_bbox = bboxes[search_idx]
        
        # Load frames
        template_frame = self.load_frame(video_path, template_bbox['frame'])
        search_frame = self.load_frame(video_path, search_bbox['frame'])
        
        # Crop and resize
        template_crop, _ = self.crop_bbox(template_frame, template_bbox)
        search_crop, crop_coords = self.crop_bbox(search_frame, search_bbox)
        
        # Convert to tensor and normalize
        template = torch.from_numpy(template_crop).permute(2, 0, 1).float() / 255.0
        search = torch.from_numpy(search_crop).permute(2, 0, 1).float() / 255.0
        
        # Normalize to [-1, 1]
        template = (template - 0.5) / 0.5
        search = (search - 0.5) / 0.5
        
        # Ground truth bbox in search frame (relative to crop)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
        gt_x1 = (search_bbox['x1'] - crop_x1) / (crop_x2 - crop_x1)
        gt_y1 = (search_bbox['y1'] - crop_y1) / (crop_y2 - crop_y1)
        gt_x2 = (search_bbox['x2'] - crop_x1) / (crop_x2 - crop_x1)
        gt_y2 = (search_bbox['y2'] - crop_y1) / (crop_y2 - crop_y1)
        
        # Convert to center format
        gt_cx = (gt_x1 + gt_x2) / 2
        gt_cy = (gt_y1 + gt_y2) / 2
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        
        gt_bbox = torch.tensor([gt_cx, gt_cy, gt_w, gt_h], dtype=torch.float32)
        
        return template, search, gt_bbox


class TrackingLoss(nn.Module):
    """Loss function cho tracking"""
    def __init__(self, cls_weight=1.0, reg_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()
    
    def forward(self, cls_pred, reg_pred, gt_bbox, anchors):
        """
        Args:
            cls_pred: [B, num_anchors, H, W]
            reg_pred: [B, num_anchors*4, H, W]
            gt_bbox: [B, 4] (cx, cy, w, h)
            anchors: [num_anchors, 4]
        """
        B, num_anchors, H, W = cls_pred.shape
        
        # Generate anchor targets
        cls_target = torch.zeros_like(cls_pred)
        reg_target = torch.zeros_like(reg_pred)
        
        # For simplicity, use positive anchor matching
        # In practice, should use IoU matching
        for b in range(B):
            gt = gt_bbox[b]
            
            # Find best matching anchor
            best_iou = 0
            best_anchor_idx = 0
            
            for a_idx in range(num_anchors):
                anchor = anchors[a_idx]
                # Simple center distance matching
                center_dist = ((gt[0] - anchor[0])**2 + (gt[1] - anchor[1])**2)**0.5
                if center_dist < 0.1:  # Threshold
                    best_anchor_idx = a_idx
                    break
            
            # Set positive anchor
            cls_target[b, best_anchor_idx, :, :] = 1.0
            
            # Regression target (relative to anchor)
            reg_target[b, best_anchor_idx*4:(best_anchor_idx+1)*4, :, :] = gt.unsqueeze(-1).unsqueeze(-1)
        
        # Losses
        cls_loss = self.cls_loss_fn(cls_pred, cls_target)
        reg_loss = self.reg_loss_fn(reg_pred, reg_target)
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return total_loss, cls_loss, reg_loss


def train_model(
    annotations_path,
    samples_dir,
    epochs=100,
    batch_size=8,
    lr=1e-3,
    device=None,
    save_dir='runs/tracking'
):
    """Train tracking model"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create model
    model = TrackingModel(num_classes=1, num_anchors=9)
    model = model.to(device)
    
    total_params = count_parameters(model)
    backbone_params = count_parameters(model.backbone)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Backbone: {backbone_params / 1e6:.2f}M")
    
    # Dataset
    train_dataset = TrackingDataset(annotations_path, samples_dir, img_size=256, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"\nDataset: {len(train_dataset)} samples")
    
    # Loss and optimizer
    criterion = TrackingLoss(cls_weight=1.0, reg_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (template, search, gt_bbox) in enumerate(pbar):
            template = template.to(device)
            search = search.to(device)
            gt_bbox = gt_bbox.to(device)
            
            # Reset hidden state for each batch
            model.reset_hidden_state()
            
            # Forward
            cls_pred, reg_pred = model(template, search, reset_hidden=True)
            
            # Generate anchors
            anchors = model.generate_anchors(
                feature_size=(cls_pred.shape[2], cls_pred.shape[3]),
                stride=16
            ).to(device)
            
            # Loss
            loss, cls_loss, reg_loss = criterion(cls_pred, reg_pred, gt_bbox, anchors)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f} (cls: {avg_cls_loss:.4f}, reg: {avg_reg_loss:.4f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_dir / 'best.pt')
            print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_dir / 'last.pt')
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Tracking Model")
    parser.add_argument("--annotations", default="observing/train/annotations/annotations.json",
                       help="Path to annotations.json")
    parser.add_argument("--samples", default="observing/train/samples",
                       help="Path to samples directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", default=None,
                       help="Device to use ('cuda' or 'cpu', None for auto-detect)")
    parser.add_argument("--save-dir", default="runs/tracking",
                       help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found!")
        exit(1)
    
    train_model(
        annotations_path=str(annotations_path),
        samples_dir=args.samples,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )

