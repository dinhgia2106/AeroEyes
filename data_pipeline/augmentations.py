"""
Module chứa các augmentation techniques theo báo cáo PHẦN 4.

Theo báo cáo, pipeline dữ liệu giải quyết 3 vấn đề chính:
1. Rò rỉ Dữ liệu (Data Leakage): Xử lý trong group_split.py
2. Vật thể nhỏ (Small Objects): Tiling/Cropping, Copy-Paste (PHẦN 4.2)
3. Khoảng cách Miền (Domain Gap): Geometric và Photometric augmentations (PHẦN 4.3)

Các augmentations này được thiết kế để:
- Giải quyết vật thể nhỏ tại training-time (thay thế SAHI)
- Thu hẹp khoảng cách miền ground-to-aerial
- Mô phỏng điều kiện ánh sáng khắc nghiệt
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import random
from pathlib import Path


class TilingAugmentation:
    """
    Tiling/Cropping augmentation để giải quyết vật thể nhỏ (PHẦN 4.2.1).
    
    Theo báo cáo: "Khai thác sức mạnh của Tiling". Thay vì luôn huấn luyện 
    trên toàn bộ khung hình (ví dụ: 1920 x 1080), bộ tải dữ liệu sẽ, với một 
    xác suất nhất định, cắt ngẫu nhiên một cửa sổ nhỏ hơn (ví dụ: 640 x 640) 
    từ ảnh gốc.
    
    Tác động: Thao tác này hoạt động như một "zoom kỹ thuật số" hiệu quả. 
    Các vật thể nhỏ, trước đây chỉ chiếm vài pixel, giờ đây chiếm một phần 
    tương đối lớn hơn của đầu vào. Điều này buộc mô hình phải học các đặc trưng 
    chi tiết của chúng, thay vì bỏ qua chúng như nhiễu.
    """
    
    def __init__(
        self,
        crop_size: Tuple[int, int] = (640, 640),
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        prob: float = 0.5
    ):
        """
        Args:
            crop_size: Kích thước cửa sổ cắt (width, height)
            min_scale: Tỷ lệ nhỏ nhất của crop so với ảnh gốc
            max_scale: Tỷ lệ lớn nhất của crop so với ảnh gốc
            prob: Xác suất áp dụng augmentation này
        """
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[Dict],  # List of {x1, y1, x2, y2}
        img_width: int,
        img_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Áp dụng tiling augmentation.
        
        Returns:
            (augmented_image, updated_bboxes)
        """
        if random.random() > self.prob:
            return image, bboxes
        
        # Chọn scale ngẫu nhiên
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_w = int(self.crop_size[0] * scale)
        crop_h = int(self.crop_size[1] * scale)
        
        # Đảm bảo crop không vượt quá kích thước ảnh và >= 1
        crop_w = max(1, min(crop_w, img_width))
        crop_h = max(1, min(crop_h, img_height))
        
        # Nếu ảnh nhỏ hơn crop_size, chỉ cần resize ảnh gốc
        if img_width <= crop_w and img_height <= crop_h:
            cropped_image = cv2.resize(image, self.crop_size)
            scale_x = self.crop_size[0] / img_width
            scale_y = self.crop_size[1] / img_height
            # Cập nhật bboxes
            updated_bboxes = []
            for bbox in bboxes:
                new_x1 = bbox['x1'] * scale_x
                new_y1 = bbox['y1'] * scale_y
                new_x2 = bbox['x2'] * scale_x
                new_y2 = bbox['y2'] * scale_y
                # Đảm bảo giá trị trong [0, crop_size]
                new_x1 = max(0, min(self.crop_size[0], new_x1))
                new_y1 = max(0, min(self.crop_size[1], new_y1))
                new_x2 = max(0, min(self.crop_size[0], new_x2))
                new_y2 = max(0, min(self.crop_size[1], new_y2))
                if new_x2 > new_x1 and new_y2 > new_y1:
                    updated_bboxes.append({
                        'x1': int(new_x1),
                        'y1': int(new_y1),
                        'x2': int(new_x2),
                        'y2': int(new_y2)
                    })
            return cropped_image, updated_bboxes
        
        # Chọn vị trí cắt ngẫu nhiên, nhưng ưu tiên vùng có bbox
        if bboxes and random.random() > 0.3:  # 70% thời gian cắt ở vùng có object
            # Chọn một bbox ngẫu nhiên làm tâm
            bbox = random.choice(bboxes)
            center_x = (bbox['x1'] + bbox['x2']) // 2
            center_y = (bbox['y1'] + bbox['y2']) // 2
        else:
            # Cắt ngẫu nhiên - đảm bảo center nằm trong phạm vi hợp lệ
            max_center_x = max(crop_w // 2, img_width - crop_w // 2 - 1)
            max_center_y = max(crop_h // 2, img_height - crop_h // 2 - 1)
            center_x = random.randint(max(0, crop_w // 2), max(crop_w // 2, max_center_x))
            center_y = random.randint(max(0, crop_h // 2), max(crop_h // 2, max_center_y))
        
        # Tính toán vùng cắt
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(img_width, x1 + crop_w)
        y2 = min(img_height, y1 + crop_h)
        
        # Điều chỉnh nếu vượt quá biên
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        # Đảm bảo vùng cắt hợp lệ
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))
        
        # Cắt ảnh
        cropped_image = image[y1:y2, x1:x2]
        
        # Kiểm tra ảnh cắt có hợp lệ không
        use_full_image = False
        if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            # Nếu ảnh cắt rỗng, trả về ảnh gốc đã resize
            cropped_image = cv2.resize(image, self.crop_size)
            scale_x = self.crop_size[0] / img_width
            scale_y = self.crop_size[1] / img_height
            use_full_image = True  # Đánh dấu sử dụng toàn bộ ảnh
        # Resize về crop_size nếu cần
        elif cropped_image.shape[0] != self.crop_size[1] or cropped_image.shape[1] != self.crop_size[0]:
            cropped_image = cv2.resize(cropped_image, self.crop_size)
            scale_x = self.crop_size[0] / (x2 - x1)
            scale_y = self.crop_size[1] / (y2 - y1)
        else:
            scale_x = 1.0
            scale_y = 1.0
        
        # Cập nhật bboxes
        updated_bboxes = []
        for bbox in bboxes:
            # Kiểm tra xem bbox có nằm trong vùng cắt không
            bbox_x1 = bbox['x1']
            bbox_y1 = bbox['y1']
            bbox_x2 = bbox['x2']
            bbox_y2 = bbox['y2']
            
            # Tính toán bbox mới trong ảnh đã cắt
            if use_full_image:
                # Nếu sử dụng toàn bộ ảnh, không cần trừ x1, y1
                new_x1 = bbox_x1 * scale_x
                new_y1 = bbox_y1 * scale_y
                new_x2 = bbox_x2 * scale_x
                new_y2 = bbox_y2 * scale_y
            else:
                # Tính toán bbox mới trong ảnh đã cắt
                new_x1 = (bbox_x1 - x1) * scale_x
                new_y1 = (bbox_y1 - y1) * scale_y
                new_x2 = (bbox_x2 - x1) * scale_x
                new_y2 = (bbox_y2 - y1) * scale_y
            
            # Đảm bảo giá trị trong [0, crop_size]
            new_x1 = max(0, min(self.crop_size[0], new_x1))
            new_y1 = max(0, min(self.crop_size[1], new_y1))
            new_x2 = max(0, min(self.crop_size[0], new_x2))
            new_y2 = max(0, min(self.crop_size[1], new_y2))
            
            if use_full_image:
                # Khi sử dụng toàn bộ ảnh, giữ lại tất cả bboxes hợp lệ
                if new_x2 > new_x1 and new_y2 > new_y1:
                    updated_bboxes.append({
                        'x1': int(new_x1),
                        'y1': int(new_y1),
                        'x2': int(new_x2),
                        'y2': int(new_y2)
                    })
            else:
                # Chỉ giữ lại bbox nếu có ít nhất 50% diện tích nằm trong vùng cắt
                bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
                if bbox_area > 0:
                    intersection_x1 = max(0, new_x1)
                    intersection_y1 = max(0, new_y1)
                    intersection_x2 = min(self.crop_size[0], new_x2)
                    intersection_y2 = min(self.crop_size[1], new_y2)
                    
                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        if intersection_area / bbox_area >= 0.5:
                            updated_bboxes.append({
                                'x1': int(intersection_x1),
                                'y1': int(intersection_y1),
                                'x2': int(intersection_x2),
                                'y2': int(intersection_y2)
                            })
        
        return cropped_image, updated_bboxes


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation để giải quyết vật thể nhỏ (PHẦN 4.2.2).
    
    Theo báo cáo: "Kỹ thuật cực kỳ hiệu quả để tăng số lượng các vật thể hiếm hoặc nhỏ."
    
    Pipeline:
    1. Sử dụng các nhãn (annotations) để cắt các pixel của một vật thể ground-truth
    2. Áp dụng các phép biến đổi hình học và màu sắc (xoay, thay đổi kích thước, độ sáng)
    3. Chọn một khung hình ngẫu nhiên khác làm hậu cảnh
    4. "Dán" vật thể này vào một vị trí hợp lý và cập nhật nhãn
    
    Tác động: Giải quyết sự mất cân bằng dữ liệu và tăng đáng kể số lượng vật thể nhỏ, 
    buộc mô hình phải học cách phát hiện chúng trong các bối cảnh đa dạng.
    """
    
    def __init__(
        self,
        prob: float = 0.3,
        max_objects: int = 2,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        rotation_range: Tuple[int, int] = (-15, 15)
    ):
        """
        Args:
            prob: Xác suất áp dụng augmentation này
            max_objects: Số lượng vật thể tối đa có thể dán vào một frame
            scale_range: Phạm vi thay đổi kích thước (scale)
            rotation_range: Phạm vi xoay (độ)
        """
        self.prob = prob
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.rotation_range = rotation_range
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        source_image: np.ndarray,
        source_bboxes: List[Dict],
        img_width: int,
        img_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Áp dụng copy-paste augmentation.
        
        Args:
            image: Ảnh đích (background)
            bboxes: Bboxes hiện có trong ảnh đích
            source_image: Ảnh nguồn chứa vật thể cần copy
            source_bboxes: Bboxes trong ảnh nguồn
            img_width, img_height: Kích thước ảnh
            
        Returns:
            (augmented_image, updated_bboxes)
        """
        if random.random() > self.prob or not source_bboxes:
            return image, bboxes
        
        augmented_image = image.copy()
        updated_bboxes = bboxes.copy()
        
        # Chọn ngẫu nhiên các vật thể từ source để dán
        num_objects = random.randint(1, min(self.max_objects, len(source_bboxes)))
        selected_source_bboxes = random.sample(source_bboxes, num_objects)
        
        for source_bbox in selected_source_bboxes:
            # Cắt vật thể từ ảnh nguồn
            x1 = max(0, int(source_bbox['x1']))
            y1 = max(0, int(source_bbox['y1']))
            x2 = min(source_image.shape[1], int(source_bbox['x2']))
            y2 = min(source_image.shape[0], int(source_bbox['y2']))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            object_patch = source_image[y1:y2, x1:x2]
            
            # Áp dụng transformations
            # 1. Scale
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_w = int(object_patch.shape[1] * scale)
            new_h = int(object_patch.shape[0] * scale)
            if new_w > 0 and new_h > 0:
                object_patch = cv2.resize(object_patch, (new_w, new_h))
            
            # 2. Rotation
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            if abs(angle) > 1:
                center = (new_w // 2, new_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                object_patch = cv2.warpAffine(object_patch, M, (new_w, new_h))
            
            # 3. Photometric transformations (nhẹ)
            if random.random() > 0.5:
                # Brightness adjustment
                brightness_delta = random.uniform(-30, 30)
                object_patch = cv2.convertScaleAbs(object_patch, alpha=1, beta=brightness_delta)
            
            # Chọn vị trí dán (tránh overlap với bboxes hiện có)
            max_attempts = 50
            for _ in range(max_attempts):
                paste_x = random.randint(0, max(1, img_width - new_w))
                paste_y = random.randint(0, max(1, img_height - new_h))
                
                # Kiểm tra overlap
                overlap = False
                new_bbox = {
                    'x1': paste_x,
                    'y1': paste_y,
                    'x2': paste_x + new_w,
                    'y2': paste_y + new_h
                }
                
                for existing_bbox in updated_bboxes:
                    if self._calculate_iou(new_bbox, existing_bbox) > 0.1:
                        overlap = True
                        break
                
                if not overlap:
                    # Dán vật thể vào ảnh
                    # Sử dụng alpha blending nếu có alpha channel, nếu không thì paste trực tiếp
                    if paste_y + new_h <= img_height and paste_x + new_w <= img_width:
                        augmented_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = object_patch
                        updated_bboxes.append(new_bbox)
                    break
        
        return augmented_image, updated_bboxes
    
    @staticmethod
    def _calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
        """Tính IoU giữa hai bboxes"""
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class GeometricAugmentation:
    """
    Geometric augmentations để thu hẹp khoảng cách miền (PHẦN 4.3.1).
    
    Theo báo cáo: "Dữ liệu drone về cơ bản là không có hướng 'lên' cố định. 
    Một fly cam bay từ phía Bắc hay phía Nam đều nhìn thấy cùng một vật thể, 
    chỉ là bị xoay đi."
    
    Các phép biến đổi bắt buộc:
    - Xoay (Rotation): Áp dụng các phép xoay 90, 180, và 270 độ
    - Lật (Flips): Áp dụng lật ngang (Horizontal Flip) và lật dọc (Vertical Flip)
    
    Tác động: Đây là các phép biến đổi thực tế (realistic transformations) cho dữ liệu 
    trên không, tăng gấp 4 lần (hoặc hơn) tập dữ liệu huấn luyện một cách hiệu quả.
    """
    
    def __init__(
        self,
        rotation_angles: List[int] = [90, 180, 270],
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        prob: float = 0.5
    ):
        """
        Args:
            rotation_angles: Danh sách các góc xoay (độ)
            flip_horizontal: Có cho phép lật ngang không
            flip_vertical: Có cho phép lật dọc không
            prob: Xác suất áp dụng augmentation này
        """
        self.rotation_angles = rotation_angles
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.prob = prob
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        img_width: int,
        img_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Áp dụng geometric augmentation.
        
        Returns:
            (augmented_image, updated_bboxes)
        """
        if random.random() > self.prob:
            return image, bboxes
        
        augmented_image = image.copy()
        updated_bboxes = bboxes.copy()
        
        # Áp dụng rotation (nếu có)
        if self.rotation_angles and random.random() > 0.5:
            angle = random.choice(self.rotation_angles)
            augmented_image, updated_bboxes = self._rotate(
                augmented_image, updated_bboxes, angle, img_width, img_height
            )
            # Cập nhật kích thước sau khi xoay
            img_height, img_width = augmented_image.shape[:2]
        
        # Áp dụng flips
        if self.flip_horizontal and random.random() > 0.5:
            augmented_image, updated_bboxes = self._flip_horizontal(
                augmented_image, updated_bboxes, img_width
            )
        
        if self.flip_vertical and random.random() > 0.5:
            augmented_image, updated_bboxes = self._flip_vertical(
                augmented_image, updated_bboxes, img_height
            )
        
        return augmented_image, updated_bboxes
    
    @staticmethod
    def _rotate(
        image: np.ndarray,
        bboxes: List[Dict],
        angle: int,
        img_width: int,
        img_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Xoay ảnh và cập nhật bboxes"""
        center = (img_width // 2, img_height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Tính toán kích thước mới sau khi xoay
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((img_height * sin) + (img_width * cos))
        new_h = int((img_height * cos) + (img_width * sin))
        
        # Điều chỉnh ma trận xoay
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
        
        # Cập nhật bboxes
        updated_bboxes = []
        for bbox in bboxes:
            # Chuyển đổi tọa độ bbox
            points = np.array([
                [bbox['x1'], bbox['y1']],
                [bbox['x2'], bbox['y1']],
                [bbox['x2'], bbox['y2']],
                [bbox['x1'], bbox['y2']]
            ], dtype=np.float32)
            
            # Áp dụng transformation
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            transformed_points = M.dot(points_ones.T).T
            
            # Tính bbox mới
            x_coords = transformed_points[:, 0]
            y_coords = transformed_points[:, 1]
            new_x1 = max(0, int(np.min(x_coords)))
            new_y1 = max(0, int(np.min(y_coords)))
            new_x2 = min(new_w, int(np.max(x_coords)))
            new_y2 = min(new_h, int(np.max(y_coords)))
            
            if new_x2 > new_x1 and new_y2 > new_y1:
                updated_bboxes.append({
                    'x1': new_x1,
                    'y1': new_y1,
                    'x2': new_x2,
                    'y2': new_y2
                })
        
        return rotated_image, updated_bboxes
    
    @staticmethod
    def _flip_horizontal(
        image: np.ndarray,
        bboxes: List[Dict],
        img_width: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Lật ngang ảnh và cập nhật bboxes"""
        flipped_image = cv2.flip(image, 1)
        
        updated_bboxes = []
        for bbox in bboxes:
            new_x1 = img_width - bbox['x2']
            new_x2 = img_width - bbox['x1']
            updated_bboxes.append({
                'x1': new_x1,
                'y1': bbox['y1'],
                'x2': new_x2,
                'y2': bbox['y2']
            })
        
        return flipped_image, updated_bboxes
    
    @staticmethod
    def _flip_vertical(
        image: np.ndarray,
        bboxes: List[Dict],
        img_height: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Lật dọc ảnh và cập nhật bboxes"""
        flipped_image = cv2.flip(image, 0)
        
        updated_bboxes = []
        for bbox in bboxes:
            new_y1 = img_height - bbox['y2']
            new_y2 = img_height - bbox['y1']
            updated_bboxes.append({
                'x1': bbox['x1'],
                'y1': new_y1,
                'x2': bbox['x2'],
                'y2': new_y2
            })
        
        return flipped_image, updated_bboxes


class PhotometricAugmentation:
    """
    Photometric augmentations để thu hẹp khoảng cách miền (PHẦN 4.3.2).
    
    Theo báo cáo: "Mô phỏng trực tiếp 'điều kiện ánh sáng khắc nghiệt' được mô tả."
    
    Các phép biến đổi bắt buộc:
    - Brightness/Contrast Jitter: Thay đổi ngẫu nhiên độ sáng và độ tương phản
    - Gamma Correction: Thay đổi gamma để mô phỏng điều kiện nắng gắt (gamma < 1) 
      hoặc thiếu sáng (gamma > 1)
    - Noise & Blur: Thêm nhiễu Gaussian và mờ chuyển động để mô phỏng nhiễu cảm biến 
      và chuyển động nhanh của drone
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (-30, 30),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        gamma_range: Tuple[float, float] = (0.7, 1.3),
        noise_std: float = 5.0,
        blur_prob: float = 0.3,
        prob: float = 0.7
    ):
        """
        Args:
            brightness_range: Phạm vi điều chỉnh độ sáng
            contrast_range: Phạm vi điều chỉnh độ tương phản
            gamma_range: Phạm vi gamma correction (gamma < 1: nắng gắt, gamma > 1: thiếu sáng)
            noise_std: Độ lệch chuẩn của nhiễu Gaussian
            blur_prob: Xác suất áp dụng blur
            prob: Xác suất áp dụng augmentation này
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.prob = prob
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Áp dụng photometric augmentation.
        
        Returns:
            (augmented_image, bboxes) - bboxes không thay đổi
        """
        if random.random() > self.prob:
            return image, bboxes
        
        augmented_image = image.copy()
        
        # Brightness adjustment
        if random.random() > 0.3:
            brightness_delta = random.uniform(*self.brightness_range)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=1, beta=brightness_delta)
        
        # Contrast adjustment
        if random.random() > 0.3:
            contrast = random.uniform(*self.contrast_range)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast, beta=0)
        
        # Gamma correction
        if random.random() > 0.3:
            gamma = random.uniform(*self.gamma_range)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            augmented_image = cv2.LUT(augmented_image, table)
        
        # Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, self.noise_std, augmented_image.shape).astype(np.float32)
            augmented_image = np.clip(augmented_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Motion blur
        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5, 7])
            # Motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            augmented_image = cv2.filter2D(augmented_image, -1, kernel)
        
        return augmented_image, bboxes

