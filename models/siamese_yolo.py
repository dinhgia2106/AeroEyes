"""
Siamese-YOLOv8 Model Architecture (PHẦN 2).

Theo báo cáo PHẦN 2, kiến trúc bao gồm:
1. Shared YOLOv8n backbone (weight sharing): Backbone xử lý video và reference images chia sẻ cùng trọng số
2. Attention Pooling: Gộp 3 ảnh tham chiếu thành query vector duy nhất bằng Multi-Head Attention
3. Custom detection head: Thay thế detection head tiêu chuẩn bằng similarity matching head
4. Multi-scale features: Sử dụng 3 scales (P3, P4, P5) từ backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv, C2f, SPPF
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Using simplified backbone.")


class AttentionPooling(nn.Module):
    """
    Attention Pooling để gộp 3 ảnh tham chiếu thành query vector (PHẦN 2.2).
    
    Theo báo cáo: "Gộp 3 ảnh tham chiếu thành query vector duy nhất bằng Multi-Head Attention,
    cho phép mô hình học cách gán trọng số cao hơn cho ảnh chất lượng tốt."
    """
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        """
        Args:
            feature_dim: Kích thước feature vector
            num_heads: Số lượng attention heads
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim phải chia hết cho num_heads"
        
        # Multi-head attention
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, ref_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_features: (B, 3, D) - 3 reference image features
        
        Returns:
            query_vector: (B, D) - Aggregated query vector
        """
        B, N, D = ref_features.shape  # N = 3 (số lượng reference images)
        
        # Self-attention: mỗi reference image là một token
        q = self.query(ref_features)  # (B, N, D)
        k = self.key(ref_features)  # (B, N, D)
        v = self.value(ref_features)  # (B, N, D)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, d)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        
        # Output projection
        output = self.output(attn_output)  # (B, N, D)
        
        # Residual connection
        output = output + ref_features
        
        # Layer norm
        output = self.norm(output)
        
        # Pool to single vector (mean pooling)
        query_vector = output.mean(dim=1)  # (B, D)
        
        return query_vector


class SimilarityHead(nn.Module):
    """
    Similarity Head để tính Cosine Similarity giữa query vector và feature map (PHẦN 2.3).
    
    Theo báo cáo: "Thay thế detection head tiêu chuẩn của YOLOv8 bằng similarity matching head,
    tính Cosine Similarity giữa query vector và mọi vị trí trên feature map."
    """
    
    def __init__(self, feature_dim: int = 256, in_channels: int = 256):
        """
        Args:
            feature_dim: Kích thước query vector
            in_channels: Số channels của feature map từ backbone
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Project feature map to feature_dim
        self.feature_proj = nn.Conv2d(in_channels, feature_dim, kernel_size=1)
        
    def forward(
        self,
        feature_map: torch.Tensor,
        query_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W) - Feature map từ backbone
            query_vector: (B, D) - Query vector từ reference images
        
        Returns:
            similarity_map: (B, 1, H, W) - Similarity scores
        """
        B, C, H, W = feature_map.shape
        
        # Project feature map
        projected_features = self.feature_proj(feature_map)  # (B, D, H, W)
        
        # Normalize query vector
        query_norm = F.normalize(query_vector, p=2, dim=1)  # (B, D)
        
        # Normalize projected features
        projected_norm = F.normalize(projected_features, p=2, dim=1)  # (B, D, H, W)
        
        # Compute cosine similarity
        # Reshape query: (B, D, 1, 1)
        query_reshaped = query_norm.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        
        # Dot product: (B, D, H, W) * (B, D, 1, 1) -> (B, 1, H, W)
        similarity = (projected_norm * query_reshaped).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        return similarity


class SimpleBackbone(nn.Module):
    """Simplified backbone nếu ultralytics không có sẵn"""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Simple CNN backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale features"""
        x1 = self.conv1(x)  # 1/2
        x2 = self.conv2(x1)  # 1/4
        x3 = self.conv3(x2)  # 1/8
        x4 = self.conv4(x3)  # 1/16
        x5 = self.conv5(x4)  # 1/32
        
        # Return P3, P4, P5 (1/8, 1/16, 1/32)
        return [x3, x4, x5]


class SiameseYOLOv8(nn.Module):
    """
    Siamese-YOLOv8 Model (PHẦN 2).
    
    Kiến trúc:
    - Shared YOLOv8n backbone (weight sharing)
    - Attention Pooling để gộp 3 reference images
    - Similarity Head để tính similarity scores
    - Multi-scale detection (P3, P4, P5)
    """
    
    def __init__(
        self,
        model_size: str = 'n',
        feature_dim: int = 256,
        num_ref_images: int = 3
    ):
        """
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            feature_dim: Kích thước feature vector
            num_ref_images: Số lượng reference images (mặc định 3)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_ref_images = num_ref_images
        
        # Backbone (shared weights)
        self.backbone_out_channels = None  # Sẽ được xác định sau
        if ULTRALYTICS_AVAILABLE:
            try:
                # Load YOLOv8 model
                yolo_model = YOLO(f'yolov8{model_size}.pt')
                # Lấy toàn bộ model để có thể forward đúng cách
                self.yolo_model = yolo_model.model
                # Extract backbone (chỉ phần backbone, không có head)
                # YOLOv8 model structure: model[0:10] là backbone
                self.backbone = self.yolo_model.model[:10]
                # Tạm thời set, sẽ được xác định chính xác sau
                self.backbone_out_channels = 256
            except Exception as e:
                print(f"Warning: Could not load YOLOv8, using simple backbone: {e}")
                self.backbone = SimpleBackbone()
                self.backbone_out_channels = 256
                self.yolo_model = None
        else:
            self.backbone = SimpleBackbone()
            self.backbone_out_channels = 256
            self.yolo_model = None
        
        # Reference image processing
        # Process each reference image through backbone
        self.ref_backbone = self.backbone  # Shared weights
        
        # Feature projection for reference images (sẽ được cập nhật sau khi biết channels)
        self.ref_feature_proj = None
        
        # Attention Pooling
        self.attention_pooling = AttentionPooling(feature_dim=feature_dim)
        
        # Detection heads (sẽ được tạo động sau khi biết số channels)
        self.objectness_heads = None
        self.bbox_heads = None
        self.similarity_heads = None
        
        # Flag để đánh dấu đã khởi tạo heads chưa
        self._heads_initialized = False
    
    def _initialize_heads(self, feature_channels: List[int], device=None):
        """Khởi tạo detection heads dựa trên số channels thực tế của features"""
        if self._heads_initialized:
            return
        
        # Xác định device
        if device is None:
            try:
                device = next(self.attention_pooling.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        
        # Khởi tạo feature projection cho reference images
        # Sử dụng channels của P5 (scale lớn nhất)
        ref_channels = feature_channels[-1]
        self.ref_feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ref_channels, self.feature_dim)
        ).to(device)
        
        # Khởi tạo detection heads cho mỗi scale
        self.objectness_heads = nn.ModuleList([
            nn.Conv2d(ch, 1, kernel_size=1) for ch in feature_channels
        ]).to(device)
        
        self.bbox_heads = nn.ModuleList([
            nn.Conv2d(ch, 4, kernel_size=1) for ch in feature_channels
        ]).to(device)
        
        self.similarity_heads = nn.ModuleList([
            SimilarityHead(feature_dim=self.feature_dim, in_channels=ch)
            for ch in feature_channels
        ]).to(device)
        
        self._heads_initialized = True
    
    def extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features từ backbone.
        
        Args:
            x: (B, 3, H, W) - Input image
        
        Returns:
            List of feature maps: [P3, P4, P5]
        """
        # Đảm bảo input là float32
        x = x.float()
        
        if ULTRALYTICS_AVAILABLE and self.yolo_model is not None:
            # Use YOLOv8 model để extract features đúng cách
            # YOLOv8 có method để trả về features
            try:
                # Forward qua model và lấy features từ các layer cụ thể
                features = []
                y = x
                
                # YOLOv8 backbone structure thường có:
                # - P3 ở layer 4-5
                # - P4 ở layer 6-7  
                # - P5 ở layer 8-9
                # Nhưng cách tốt nhất là forward qua toàn bộ backbone và lấy output
                
                # Forward qua backbone
                for i, layer in enumerate(self.backbone):
                    y = layer(y)
                    # Lấy features ở các điểm quan trọng
                    # Thường là sau các C2f blocks
                    if i in [4, 6, 9]:  # Các điểm có thể là P3, P4, P5
                        features.append(y)
                
                # Nếu không đủ 3 scales, lấy thêm từ các layer cuối
                if len(features) < 3:
                    # Lấy thêm từ layer cuối cùng
                    while len(features) < 3:
                        features.append(y)
                
                # Đảm bảo có đúng 3 scales
                if len(features) > 3:
                    features = features[-3:]  # Lấy 3 scales cuối
                
                # Khởi tạo heads nếu chưa
                if not self._heads_initialized:
                    feature_channels = [f.shape[1] for f in features]
                    device = features[0].device if len(features) > 0 else None
                    self._initialize_heads(feature_channels, device)
                
                return features[:3]
            except Exception as e:
                print(f"Warning: Error extracting YOLOv8 features: {e}")
                # Fallback to simple backbone
                return self._extract_simple_backbone(x)
        else:
            return self._extract_simple_backbone(x)
    
    def _extract_simple_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features từ simple backbone"""
        features = self.backbone(x)
        # Khởi tạo heads nếu chưa
        if not self._heads_initialized:
            feature_channels = [f.shape[1] for f in features]
            device = features[0].device if len(features) > 0 else x.device
            self._initialize_heads(feature_channels, device)
        return features
    
    def extract_reference_features(self, reference_images: torch.Tensor) -> torch.Tensor:
        """
        Extract và gộp features từ reference images.
        
        Args:
            reference_images: (B, N, 3, H, W) - N reference images
        
        Returns:
            query_vector: (B, D) - Aggregated query vector
        """
        # Đảm bảo input là float32
        reference_images = reference_images.float()
        
        B, N, C, H, W = reference_images.shape
        
        # Reshape để xử lý batch
        ref_flat = reference_images.view(B * N, C, H, W)
        
        # Extract features từ backbone
        ref_features_list = self.extract_backbone_features(ref_flat)
        # Lấy P5 (scale lớn nhất)
        ref_features = ref_features_list[-1]  # (B*N, C, H', W')
        
        # Đảm bảo ref_feature_proj đã được khởi tạo
        if self.ref_feature_proj is None:
            feature_channels = [f.shape[1] for f in ref_features_list]
            device = ref_features.device
            self._initialize_heads(feature_channels, device)
        
        # Project to feature_dim
        ref_features_vec = self.ref_feature_proj(ref_features)  # (B*N, D)
        
        # Reshape back
        ref_features_vec = ref_features_vec.view(B, N, self.feature_dim)  # (B, N, D)
        
        # Attention pooling
        query_vector = self.attention_pooling(ref_features_vec)  # (B, D)
        
        return query_vector
    
    def forward(
        self,
        video_frames: torch.Tensor,
        reference_images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            video_frames: (B, 3, H, W) - Video frames
            reference_images: (B, N, 3, H, W) - N reference images
        
        Returns:
            (objectness_preds, similarity_preds, bbox_preds)
            Mỗi là List of tensors cho 3 scales (P3, P4, P5)
        """
        # Đảm bảo inputs là float32
        video_frames = video_frames.float()
        reference_images = reference_images.float()
        
        # Extract query vector từ reference images
        query_vector = self.extract_reference_features(reference_images)  # (B, D)
        
        # Extract features từ video frames
        video_features = self.extract_backbone_features(video_frames)  # List of (B, C, H, W)
        
        # Đảm bảo heads đã được khởi tạo
        if not self._heads_initialized:
            feature_channels = [f.shape[1] for f in video_features]
            device = video_features[0].device if len(video_features) > 0 else video_frames.device
            self._initialize_heads(feature_channels, device)
        
        # Apply detection heads
        objectness_preds = []
        similarity_preds = []
        bbox_preds = []
        
        for i, (feat, obj_head, sim_head, bbox_head) in enumerate(
            zip(video_features, self.objectness_heads, self.similarity_heads, self.bbox_heads)
        ):
            # Objectness prediction
            obj_pred = obj_head(feat)  # (B, 1, H, W)
            objectness_preds.append(obj_pred)
            
            # Similarity prediction
            sim_pred = sim_head(feat, query_vector)  # (B, 1, H, W)
            similarity_preds.append(sim_pred)
            
            # Bbox prediction
            bbox_pred = bbox_head(feat)  # (B, 4, H, W)
            bbox_preds.append(bbox_pred)
        
        return objectness_preds, similarity_preds, bbox_preds


if __name__ == "__main__":
    # Test model
    print("Testing SiameseYOLOv8 Model...")
    
    model = SiameseYOLOv8(model_size='n', feature_dim=256, num_ref_images=3)
    
    # Test forward pass
    B = 2
    video_frames = torch.randn(B, 3, 640, 640)
    reference_images = torch.randn(B, 3, 3, 640, 640)
    
    obj_preds, sim_preds, bbox_preds = model(video_frames, reference_images)
    
    print(f"Input video_frames: {video_frames.shape}")
    print(f"Input reference_images: {reference_images.shape}")
    print(f"Output objectness_preds: {[p.shape for p in obj_preds]}")
    print(f"Output similarity_preds: {[p.shape for p in sim_preds]}")
    print(f"Output bbox_preds: {[p.shape for p in bbox_preds]}")
    
    # Test extract methods
    query_vector = model.extract_reference_features(reference_images)
    print(f"Query vector: {query_vector.shape}")
    
    video_features = model.extract_backbone_features(video_frames)
    print(f"Video features: {[f.shape for f in video_features]}")
    
    print("✅ Model tested successfully!")


