"""
Module để thực hiện Group-Based Splitting (chia theo video_id) để tránh data leakage.

Thay vì chia ngẫu nhiên các frames, chúng ta chia toàn bộ video_id thành train/val,
đảm bảo tập validation bao gồm các video hoàn toàn chưa từng thấy.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random


def extract_video_ids(annotations_path: str) -> List[str]:
    """
    Trích xuất danh sách tất cả các video_id duy nhất từ annotations.json.
    
    Args:
        annotations_path: Đường dẫn đến file annotations.json
        
    Returns:
        Danh sách các video_id (ví dụ: ["Backpack_0", "Backpack_1", ...])
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    video_ids = [ann['video_id'] for ann in annotations]
    return sorted(list(set(video_ids)))  # Loại bỏ trùng lặp và sắp xếp


def group_based_split(
    annotations_path: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Thực hiện group-based splitting: chia danh sách video_id thành train/val.
    
    Args:
        annotations_path: Đường dẫn đến file annotations.json
        train_ratio: Tỷ lệ video cho tập train (mặc định 0.8 = 80%)
        random_seed: Seed cho random để đảm bảo reproducibility
        
    Returns:
        Tuple (train_video_ids, val_video_ids)
    """
    video_ids = extract_video_ids(annotations_path)
    
    # Set random seed để đảm bảo reproducibility
    random.seed(random_seed)
    video_ids_shuffled = video_ids.copy()
    random.shuffle(video_ids_shuffled)
    
    # Chia thành train và val
    num_train = int(len(video_ids_shuffled) * train_ratio)
    train_video_ids = sorted(video_ids_shuffled[:num_train])
    val_video_ids = sorted(video_ids_shuffled[num_train:])
    
    return train_video_ids, val_video_ids


def get_split_for_video(
    video_id: str,
    train_video_ids: List[str],
    val_video_ids: List[str]
) -> str:
    """
    Xác định một video_id thuộc split nào (train hay val).
    
    Args:
        video_id: ID của video cần kiểm tra
        train_video_ids: Danh sách video_id thuộc tập train
        val_video_ids: Danh sách video_id thuộc tập val
        
    Returns:
        "train" hoặc "val"
    """
    if video_id in train_video_ids:
        return "train"
    elif video_id in val_video_ids:
        return "val"
    else:
        raise ValueError(f"Video ID '{video_id}' không có trong train hoặc val splits")


if __name__ == "__main__":
    # Test script
    annotations_path = "observing/train/annotations/annotations.json"
    
    train_ids, val_ids = group_based_split(annotations_path, train_ratio=0.8)
    
    print(f"Tổng số video: {len(train_ids) + len(val_ids)}")
    print(f"Train videos ({len(train_ids)}): {train_ids}")
    print(f"Val videos ({len(val_ids)}): {val_ids}")


