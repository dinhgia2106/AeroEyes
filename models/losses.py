"""
Multi-task Loss Functions cho Siamese-YOLOv8 (PHẦN 6).

Theo báo cáo PHẦN 6, hàm mất mát tổng thể là tổng có trọng số của hai thành phần:
L_total = w1 * L_detection + w2 * L_similarity

Bao gồm:
1. Detection Loss (PHẦN 6.2): CIoU Loss + Focal Loss
   - Box Loss: CIoU (Complete IoU) hoặc SIoU để tối ưu hóa vị trí (x, y, w, h)
   - Objectness Loss: Focal Loss hoặc Binary Cross-Entropy (BCE)
   
2. Similarity Loss (PHẦN 6.3): Triplet Loss
   - Mục tiêu: Distance(A, P) + margin < Distance(A, N)
   - Anchor (A): Vector đặc trưng từ aerial view (ground-truth bbox)
   - Positive (P): Query vector Q_v từ reference images
   - Negative (N): Vector đặc trưng từ hard negative objects
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class CIoULoss(nn.Module):
    """
    Complete IoU Loss cho bounding box regression (PHẦN 6.2).
    
    Theo báo cáo: "Sử dụng một hàm mất mát bounding box hiện đại như CIoU 
    (Complete IoU) hoặc SIoU để tối ưu hóa vị trí (x, y, w, h)."
    """
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) format (x_center, y_center, width, height)
            target_boxes: (N, 4) format (x_center, y_center, width, height)
        
        Returns:
            CIoU loss: scalar
        """
        # Convert to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        
        # Center distance
        pred_center_x = pred_boxes[:, 0]
        pred_center_y = pred_boxes[:, 1]
        target_center_x = target_boxes[:, 0]
        target_center_y = target_boxes[:, 1]
        
        center_dist_sq = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # Aspect ratio consistency
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + self.eps)) -
            torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + self.eps)), 2
        )
        alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        cious = iou - (center_dist_sq / (enclose_diag_sq + self.eps)) - alpha * v
        
        return 1 - cious.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss cho objectness classification (PHẦN 6.2).
    
    Theo báo cáo: "Sử dụng Focal Loss hoặc Binary Cross-Entropy (BCE) để huấn luyện 
    objectness_score (phân biệt vật thể với hậu cảnh)."
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits, shape (N,)
            target: Target labels (0 or 1), shape (N,)
        
        Returns:
            Focal loss: scalar
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Focal term
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        
        # Weighted focal loss
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_term * bce_loss
        
        return focal_loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet Loss cho similarity learning (PHẦN 6.3).
    
    Theo báo cáo: "Đây là thành phần quan trọng nhất để giải quyết Khoảng cách Miền 
    (Ràng buộc 3). Triplet Loss tối ưu hóa thứ hạng tương đối (relative ranking) và 
    tập trung vào các 'hard negatives' (mẫu âm khó)."
    
    Triển khai "Triplet Mining":
    - Anchor (A): Vector đặc trưng của ground-truth bounding box trong khung hình video
    - Positive (P): Query vector Q_v đã được tính toán của vật thể đó
    - Negative (N): Vector đặc trưng của một "hard negative" – vật thể khác
    
    Mục tiêu Loss: Distance(A, P) + margin < Distance(A, N)
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        """
        Args:
            margin: Margin cho triplet loss
            distance_metric: 'euclidean' hoặc 'cosine'
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def compute_distance(
        self,
        anchor: torch.Tensor,
        other: torch.Tensor
    ) -> torch.Tensor:
        """
        Tính khoảng cách giữa anchor và other.
        
        Args:
            anchor: (N, D) feature vectors
            other: (N, D) feature vectors
        
        Returns:
            Distances: (N,)
        """
        if self.distance_metric == 'euclidean':
            return torch.norm(anchor - other, p=2, dim=1)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            other_norm = F.normalize(other, p=2, dim=1)
            cosine_sim = (anchor_norm * other_norm).sum(dim=1)
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: Feature vectors từ aerial view (ground-truth bbox), shape (N, D)
            positive: Query vectors từ reference images, shape (N, D)
            negative: Feature vectors từ negative objects, shape (N, D)
        
        Returns:
            Triplet loss: scalar
        """
        # Compute distances
        dist_pos = self.compute_distance(anchor, positive)  # (N,)
        dist_neg = self.compute_distance(anchor, negative)  # (N,)
        
        # Triplet loss: max(0, dist_pos - dist_neg + margin)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined Multi-task Loss: Detection Loss + Similarity Loss (PHẦN 6.1).
    
    Theo báo cáo: "Hàm mất mát tổng thể sẽ là tổng có trọng số của hai thành phần:
    L_total = w1 * L_detection + w2 * L_similarity
    
    trong đó w1 và w2 là các siêu tham số (hyperparameters) để cân bằng hai nhiệm vụ."
    """
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        similarity_weight: float = 0.5,
        triplet_margin: float = 1.0
    ):
        """
        Args:
            detection_weight: Weight cho detection loss (w1)
            similarity_weight: Weight cho similarity loss (w2)
            triplet_margin: Margin cho triplet loss
        """
        super().__init__()
        self.detection_weight = detection_weight
        self.similarity_weight = similarity_weight
        
        # Loss components
        self.ciou_loss = CIoULoss()
        self.focal_loss = FocalLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin)
    
    def forward(
        self,
        # Detection predictions
        objectness_preds: List[torch.Tensor],
        similarity_preds: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        # Detection targets
        objectness_targets: List[torch.Tensor],
        bbox_targets: List[torch.Tensor],
        # Similarity targets (for triplet loss)
        anchor_features: Optional[torch.Tensor] = None,
        positive_features: Optional[torch.Tensor] = None,
        negative_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            objectness_preds: List of predicted objectness scores
            similarity_preds: List of predicted similarity scores
            bbox_preds: List of predicted bounding boxes
            objectness_targets: List of target objectness labels
            bbox_targets: List of target bounding boxes
            anchor_features: Feature vectors từ aerial view (N, D)
            positive_features: Query vectors (N, D)
            negative_features: Negative feature vectors (N, D)
        
        Returns:
            Total loss và loss breakdown dictionary
        """
        # Detection Loss
        detection_loss = 0.0
        num_detections = 0
        
        for obj_pred, sim_pred, bbox_pred, obj_target, bbox_target in zip(
            objectness_preds, similarity_preds, bbox_preds,
            objectness_targets, bbox_targets
        ):
            # Flatten predictions
            obj_pred_flat = obj_pred.view(-1)
            bbox_pred_flat = bbox_pred.view(-1, 4)
            obj_target_flat = obj_target.view(-1)
            bbox_target_flat = bbox_target.view(-1, 4)
            
            # Mask for positive samples (where objectness > 0)
            pos_mask = obj_target_flat > 0.5
            
            if pos_mask.sum() > 0:
                # Objectness loss (Focal Loss)
                obj_loss = self.focal_loss(obj_pred_flat, obj_target_flat)
                
                # Bbox loss (CIoU) - chỉ cho positive samples
                pos_bbox_pred = bbox_pred_flat[pos_mask]
                pos_bbox_target = bbox_target_flat[pos_mask]
                bbox_loss = self.ciou_loss(pos_bbox_pred, pos_bbox_target)
                
                detection_loss += obj_loss + bbox_loss
                num_detections += 1
        
        if num_detections > 0:
            detection_loss = detection_loss / num_detections
        
        # Similarity Loss (Triplet Loss)
        similarity_loss = torch.tensor(0.0, device=objectness_preds[0].device)
        
        if anchor_features is not None and positive_features is not None and negative_features is not None:
            similarity_loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        
        # Total loss
        total_loss = (
            self.detection_weight * detection_loss +
            self.similarity_weight * similarity_loss
        )
        
        # Loss breakdown
        loss_dict = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'similarity_loss': similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    print("Testing Loss Functions...")
    
    # CIoU Loss
    pred_boxes = torch.tensor([[100, 100, 50, 50], [200, 200, 60, 60]], dtype=torch.float32)
    target_boxes = torch.tensor([[105, 105, 50, 50], [195, 195, 60, 60]], dtype=torch.float32)
    ciou = CIoULoss()
    loss_ciou = ciou(pred_boxes, target_boxes)
    print(f"CIoU Loss: {loss_ciou.item():.4f}")
    
    # Focal Loss
    pred_logits = torch.randn(10)
    target_labels = torch.randint(0, 2, (10,)).float()
    focal = FocalLoss()
    loss_focal = focal(pred_logits, target_labels)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    
    # Triplet Loss
    anchor = torch.randn(5, 256)
    positive = torch.randn(5, 256)
    negative = torch.randn(5, 256)
    triplet = TripletLoss(margin=1.0)
    loss_triplet = triplet(anchor, positive, negative)
    print(f"Triplet Loss: {loss_triplet.item():.4f}")
    
    print("✅ All loss functions tested successfully!")

