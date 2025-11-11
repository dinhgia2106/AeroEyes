"""
Triplet Mining Strategy cho Triplet Loss (PHẦN 6.3).

Theo báo cáo: "Triển khai 'Triplet Mining' (Khai thác Bộ ba):
- Anchor (A): Vector đặc trưng của ground-truth bounding box trong khung hình video
- Positive (P): Query vector Q_v đã được tính toán của vật thể đó
- Negative (N): Vector đặc trưng của một 'hard negative' – vật thể khác"

Module này triển khai hard negative mining để tìm các negative samples
khó nhất (hardest negatives) cho triplet loss training. Hard negatives là các
samples mà mô hình dễ nhầm lẫn, giúp cải thiện khả năng phân biệt của mô hình.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import random


class TripletMiner:
    """
    Triplet Miner để tìm anchor, positive, và hard negative samples (PHẦN 6.3).
    
    Theo báo cáo: "Sử dụng các kỹ thuật 'hard negative mining' (khai thác mẫu âm khó) 
    và tinh chỉnh cẩn thận siêu tham số margin (lề)."
    
    Hard negative mining giúp mô hình học cách phân biệt các vật thể tương tự nhau,
    ví dụ: "Backpack_0" (mục tiêu) với "Backpack_1" (vật thể khác cùng loại).
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        hard_negative_ratio: float = 0.5
    ):
        """
        Args:
            margin: Margin cho triplet loss
            hard_negative_ratio: Tỷ lệ hard negatives trong batch
        """
        self.margin = margin
        self.hard_negative_ratio = hard_negative_ratio
    
    def mine_triplets(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        distance_metric: str = 'euclidean'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine triplets với hard negative mining.
        
        Args:
            anchor_features: (N, D) feature vectors từ aerial view
            positive_features: (N, D) query vectors từ reference images
            negative_features: (M, D) feature vectors từ negative objects (M >= N)
            distance_metric: 'euclidean' hoặc 'cosine'
        
        Returns:
            (selected_anchors, selected_positives, selected_negatives)
            Mỗi tensor có shape (K, D) với K <= N
        """
        N, D = anchor_features.shape
        M = negative_features.shape[0]
        
        # Compute distances
        if distance_metric == 'euclidean':
            dist_pos = torch.norm(anchor_features - positive_features, p=2, dim=1)  # (N,)
            dist_neg = torch.cdist(anchor_features, negative_features, p=2)  # (N, M)
        elif distance_metric == 'cosine':
            anchor_norm = F.normalize(anchor_features, p=2, dim=1)
            pos_norm = F.normalize(positive_features, p=2, dim=1)
            neg_norm = F.normalize(negative_features, p=2, dim=1)
            
            dist_pos = 1 - (anchor_norm * pos_norm).sum(dim=1)  # (N,)
            dist_neg = 1 - torch.matmul(anchor_norm, neg_norm.T)  # (N, M)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Hard negative mining: Tìm negative gần anchor nhất nhưng vẫn xa hơn positive
        # Hard negative là negative mà: dist(anchor, negative) < dist(anchor, positive) + margin
        # Nhưng chúng ta muốn chọn negative gần anchor nhất (hardest)
        
        # Tìm hardest negative cho mỗi anchor
        hardest_neg_indices = torch.argmin(dist_neg, dim=1)  # (N,) - index của negative gần nhất
        
        # Filter triplets: Chỉ giữ lại các triplets "valid" (có thể học được)
        # Valid triplet: dist_pos < dist_neg + margin (tức là chưa đạt được mục tiêu)
        valid_mask = []
        selected_anchors = []
        selected_positives = []
        selected_negatives = []
        
        for i in range(N):
            pos_dist = dist_pos[i].item()
            neg_idx = hardest_neg_indices[i].item()
            neg_dist = dist_neg[i, neg_idx].item()
            
            # Check if triplet is valid (hard negative)
            if neg_dist < pos_dist + self.margin:
                valid_mask.append(True)
                selected_anchors.append(anchor_features[i])
                selected_positives.append(positive_features[i])
                selected_negatives.append(negative_features[neg_idx])
            else:
                valid_mask.append(False)
        
        if len(selected_anchors) == 0:
            # Nếu không có hard negatives, chọn random negatives
            num_samples = min(N, M)
            indices = random.sample(range(N), num_samples)
            neg_indices = random.sample(range(M), num_samples)
            
            selected_anchors = anchor_features[indices]
            selected_positives = positive_features[indices]
            selected_negatives = negative_features[neg_indices]
        else:
            selected_anchors = torch.stack(selected_anchors)
            selected_positives = torch.stack(selected_positives)
            selected_negatives = torch.stack(selected_negatives)
        
        return selected_anchors, selected_positives, selected_negatives
    
    def mine_semi_hard_negatives(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        distance_metric: str = 'euclidean'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine semi-hard negatives (dễ hơn hard negatives nhưng vẫn có thể học được).
        
        Semi-hard negative: dist(anchor, positive) < dist(anchor, negative) < dist(anchor, positive) + margin
        """
        N, D = anchor_features.shape
        M = negative_features.shape[0]
        
        # Compute distances
        if distance_metric == 'euclidean':
            dist_pos = torch.norm(anchor_features - positive_features, p=2, dim=1)  # (N,)
            dist_neg = torch.cdist(anchor_features, negative_features, p=2)  # (N, M)
        elif distance_metric == 'cosine':
            anchor_norm = F.normalize(anchor_features, p=2, dim=1)
            pos_norm = F.normalize(positive_features, p=2, dim=1)
            neg_norm = F.normalize(negative_features, p=2, dim=1)
            
            dist_pos = 1 - (anchor_norm * pos_norm).sum(dim=1)
            dist_neg = 1 - torch.matmul(anchor_norm, neg_norm.T)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        selected_anchors = []
        selected_positives = []
        selected_negatives = []
        
        for i in range(N):
            pos_dist = dist_pos[i].item()
            
            # Tìm semi-hard negatives
            semi_hard_mask = (dist_neg[i] > pos_dist) & (dist_neg[i] < pos_dist + self.margin)
            
            if semi_hard_mask.any():
                # Chọn random semi-hard negative
                semi_hard_indices = torch.where(semi_hard_mask)[0]
                neg_idx = random.choice(semi_hard_indices.tolist())
                
                selected_anchors.append(anchor_features[i])
                selected_positives.append(positive_features[i])
                selected_negatives.append(negative_features[neg_idx])
            else:
                # Fallback to hardest negative
                neg_idx = torch.argmin(dist_neg[i]).item()
                selected_anchors.append(anchor_features[i])
                selected_positives.append(positive_features[i])
                selected_negatives.append(negative_features[neg_idx])
        
        if len(selected_anchors) > 0:
            return (
                torch.stack(selected_anchors),
                torch.stack(selected_positives),
                torch.stack(selected_negatives)
            )
        else:
            # Fallback
            num_samples = min(N, M)
            indices = random.sample(range(N), num_samples)
            neg_indices = random.sample(range(M), num_samples)
            
            return (
                anchor_features[indices],
                positive_features[indices],
                negative_features[neg_indices]
            )

