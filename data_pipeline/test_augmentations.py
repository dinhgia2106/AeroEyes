"""
Script test để kiểm tra các augmentations hoạt động đúng.
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from data_pipeline.augmentations import (
    TilingAugmentation,
    CopyPasteAugmentation,
    GeometricAugmentation,
    PhotometricAugmentation
)


def test_tiling_augmentation():
    """Test Tiling augmentation"""
    print("🧪 Testing Tiling Augmentation...")
    
    # Tạo ảnh test
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    img_width, img_height = 1920, 1080
    
    # Tạo bbox test (vật thể nhỏ ở góc trên bên trái)
    bboxes = [{
        'x1': 100,
        'y1': 50,
        'x2': 200,
        'y2': 150
    }]
    
    tiling_aug = TilingAugmentation(crop_size=(640, 640), prob=1.0)
    aug_img, aug_bboxes = tiling_aug(img, bboxes, img_width, img_height)
    
    print(f"   Original size: {img.shape}")
    print(f"   Augmented size: {aug_img.shape}")
    print(f"   Original bboxes: {len(bboxes)}")
    print(f"   Augmented bboxes: {len(aug_bboxes)}")
    
    # Vẽ bbox lên ảnh
    for bbox in aug_bboxes:
        cv2.rectangle(aug_img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
    
    cv2.imwrite("test_tiling_output.jpg", aug_img)
    print("   ✅ Saved test_tiling_output.jpg")


def test_geometric_augmentation():
    """Test Geometric augmentation"""
    print("\n🧪 Testing Geometric Augmentation...")
    
    # Tạo ảnh test với bbox
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_width, img_height = 640, 640
    
    bboxes = [{
        'x1': 200,
        'y1': 200,
        'x2': 400,
        'y2': 400
    }]
    
    geometric_aug = GeometricAugmentation(prob=1.0)
    aug_img, aug_bboxes = geometric_aug(img, bboxes, img_width, img_height)
    
    print(f"   Original size: {img.shape}")
    print(f"   Augmented size: {aug_img.shape}")
    print(f"   Original bboxes: {len(bboxes)}")
    print(f"   Augmented bboxes: {len(aug_bboxes)}")
    
    # Vẽ bbox
    for bbox in aug_bboxes:
        cv2.rectangle(aug_img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
    
    cv2.imwrite("test_geometric_output.jpg", aug_img)
    print("   ✅ Saved test_geometric_output.jpg")


def test_photometric_augmentation():
    """Test Photometric augmentation"""
    print("\n🧪 Testing Photometric Augmentation...")
    
    # Tạo ảnh test
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    bboxes = [{
        'x1': 200,
        'y1': 200,
        'x2': 400,
        'y2': 400
    }]
    
    photometric_aug = PhotometricAugmentation(prob=1.0)
    aug_img, aug_bboxes = photometric_aug(img, bboxes)
    
    print(f"   Original shape: {img.shape}")
    print(f"   Augmented shape: {aug_img.shape}")
    print(f"   Bboxes unchanged: {len(bboxes) == len(aug_bboxes)}")
    
    cv2.imwrite("test_photometric_output.jpg", aug_img)
    print("   ✅ Saved test_photometric_output.jpg")


def test_group_split():
    """Test Group-based splitting"""
    print("\n🧪 Testing Group-Based Splitting...")
    
    from data_pipeline.group_split import group_based_split, extract_video_ids
    
    annotations_path = "observing/train/annotations/annotations.json"
    
    if not Path(annotations_path).exists():
        print(f"   ⚠️  File không tồn tại: {annotations_path}")
        return
    
    video_ids = extract_video_ids(annotations_path)
    print(f"   Total video IDs: {len(video_ids)}")
    
    train_ids, val_ids = group_based_split(annotations_path, train_ratio=0.8)
    
    print(f"   Train videos: {len(train_ids)}")
    print(f"   Val videos: {len(val_ids)}")
    print(f"   ✅ No overlap: {len(set(train_ids) & set(val_ids)) == 0}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING DATA PIPELINE AUGMENTATIONS")
    print("=" * 60)
    
    test_tiling_augmentation()
    test_geometric_augmentation()
    test_photometric_augmentation()
    test_group_split()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


