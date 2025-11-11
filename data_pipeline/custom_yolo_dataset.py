"""
Custom YOLO Dataset với Augmentations on-the-fly.

Module này tạo một custom dataset cho YOLOv8 với khả năng áp dụng
các augmentations trong quá trình training, thay vì pre-process.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

from data_pipeline.augmentations import (
    TilingAugmentation,
    CopyPasteAugmentation,
    GeometricAugmentation,
    PhotometricAugmentation
)


class AugmentedYOLODataset:
    """
    Custom dataset cho YOLO với augmentations on-the-fly.
    
    Dataset này đọc ảnh và labels từ thư mục YOLO format và áp dụng
    augmentations mỗi khi lấy sample.
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: Tuple[int, int] = (640, 640),
        apply_augmentations: bool = True,
        tiling_prob: float = 0.5,
        geometric_prob: float = 0.5,
        photometric_prob: float = 0.7,
        copy_paste_prob: float = 0.2
    ):
        """
        Args:
            images_dir: Thư mục chứa ảnh
            labels_dir: Thư mục chứa labels (.txt files)
            img_size: Kích thước ảnh đầu vào (width, height)
            apply_augmentations: Có áp dụng augmentations không
            tiling_prob: Xác suất áp dụng tiling augmentation
            geometric_prob: Xác suất áp dụng geometric augmentation
            photometric_prob: Xác suất áp dụng photometric augmentation
            copy_paste_prob: Xác suất áp dụng copy-paste augmentation
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.apply_augmentations = apply_augmentations
        
        # Khởi tạo augmentations
        if apply_augmentations:
            self.tiling_aug = TilingAugmentation(
                crop_size=img_size,
                prob=tiling_prob
            )
            self.geometric_aug = GeometricAugmentation(prob=geometric_prob)
            self.photometric_aug = PhotometricAugmentation(prob=photometric_prob)
            self.copy_paste_aug = CopyPasteAugmentation(prob=copy_paste_prob)
        
        # Load danh sách ảnh và labels
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.labels_dir.glob("*.txt")))
        
        # Tạo mapping: image_stem -> (image_path, label_path)
        self.samples = []
        for img_path in self.image_files:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.samples.append((img_path, label_path))
        
        print(f"📦 Loaded {len(self.samples)} samples from {images_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict], str]:
        """
        Lấy một sample từ dataset.
        
        Returns:
            (image, bboxes, image_path)
            - image: np.ndarray (H, W, 3) BGR
            - bboxes: List[Dict] với keys ['x1', 'y1', 'x2', 'y2']
            - image_path: str path to image
        """
        img_path, label_path = self.samples[idx]
        
        # Đọc ảnh
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
        
        original_h, original_w = image.shape[:2]
        
        # Đọc labels
        bboxes = self._read_yolo_labels(label_path, original_w, original_h)
        
        # Áp dụng augmentations
        if self.apply_augmentations:
            image, bboxes = self._apply_augmentations(image, bboxes, original_w, original_h)
        
        # Resize về img_size nếu cần
        current_h, current_w = image.shape[:2]
        if current_w != self.img_size[0] or current_h != self.img_size[1]:
            image, bboxes = self._resize_with_bboxes(
                image, bboxes, self.img_size[0], self.img_size[1]
            )
        
        return image, bboxes, str(img_path)
    
    def _read_yolo_labels(
        self,
        label_path: Path,
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """
        Đọc labels từ file YOLO format và convert sang (x1, y1, x2, y2).
        
        Args:
            label_path: Đường dẫn đến file label
            img_width: Chiều rộng ảnh
            img_height: Chiều cao ảnh
            
        Returns:
            List[Dict] với keys ['x1', 'y1', 'x2', 'y2']
        """
        bboxes = []
        
        if not label_path.exists():
            return bboxes
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO format: class x_center y_center width height (normalized)
                cls, x_center, y_center, width, height = map(float, parts[:5])
                
                # Convert về pixel coordinates
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                # Đảm bảo trong phạm vi ảnh
                x1 = max(0, min(img_width - 1, x1))
                y1 = max(0, min(img_height - 1, y1))
                x2 = max(0, min(img_width - 1, x2))
                y2 = max(0, min(img_height - 1, y2))
                
                if x2 > x1 and y2 > y1:
                    bboxes.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
        
        return bboxes
    
    def _apply_augmentations(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        img_width: int,
        img_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Áp dụng tất cả augmentations"""
        # 1. Tiling (có thể thay đổi kích thước ảnh)
        if random.random() < self.tiling_aug.prob:
            image, bboxes = self.tiling_aug(image, bboxes, img_width, img_height)
            img_height, img_width = image.shape[:2]
        
        # 2. Geometric augmentations
        image, bboxes = self.geometric_aug(image, bboxes, img_width, img_height)
        img_height, img_width = image.shape[:2]
        
        # 3. Photometric augmentations
        image, bboxes = self.photometric_aug(image, bboxes)
        
        # 4. Copy-Paste (cần source image - sẽ được xử lý riêng nếu cần)
        # Note: Copy-Paste yêu cầu source image, nên có thể cần implement riêng
        
        return image, bboxes
    
    @staticmethod
    def _resize_with_bboxes(
        image: np.ndarray,
        bboxes: List[Dict],
        target_width: int,
        target_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resize ảnh và cập nhật bboxes tương ứng.
        
        Args:
            image: Ảnh gốc
            bboxes: List bboxes
            target_width: Chiều rộng mục tiêu
            target_height: Chiều cao mục tiêu
            
        Returns:
            (resized_image, updated_bboxes)
        """
        original_h, original_w = image.shape[:2]
        
        # Resize ảnh
        resized_image = cv2.resize(image, (target_width, target_height))
        
        # Tính scale factors
        scale_x = target_width / original_w
        scale_y = target_height / original_h
        
        # Cập nhật bboxes
        updated_bboxes = []
        for bbox in bboxes:
            new_x1 = int(bbox['x1'] * scale_x)
            new_y1 = int(bbox['y1'] * scale_y)
            new_x2 = int(bbox['x2'] * scale_x)
            new_y2 = int(bbox['y2'] * scale_y)
            
            # Đảm bảo trong phạm vi
            new_x1 = max(0, min(target_width - 1, new_x1))
            new_y1 = max(0, min(target_height - 1, new_y1))
            new_x2 = max(0, min(target_width - 1, new_x2))
            new_y2 = max(0, min(target_height - 1, new_y2))
            
            if new_x2 > new_x1 and new_y2 > new_y1:
                updated_bboxes.append({
                    'x1': new_x1,
                    'y1': new_y1,
                    'x2': new_x2,
                    'y2': new_y2
                })
        
        return resized_image, updated_bboxes
    
    def get_all_samples_for_copy_paste(self) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        Lấy tất cả samples để sử dụng cho Copy-Paste augmentation.
        
        Returns:
            List[(image, bboxes)]
        """
        samples = []
        for img_path, label_path in self.samples[:100]:  # Giới hạn để tiết kiệm memory
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            bboxes = self._read_yolo_labels(label_path, w, h)
            
            if bboxes:  # Chỉ lấy samples có bboxes
                samples.append((image, bboxes))
        
        return samples


