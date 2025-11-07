"""
Mô hình Object Tracking với:
- Backbone: CSPNet-Tiny (~6.2M params)
- Neck: BiFPN-Lite
- Temporal: ConvGRU (học được)
- Head: Depthwise Cross-Correlation + RPN (9 anchors)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== CSPNet-Tiny Backbone ====================
class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """CSP (Cross Stage Partial) Block"""
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Main branch
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNReLU(hidden_channels, hidden_channels, 3)
        self.conv3 = ConvBNReLU(hidden_channels, hidden_channels, 3)
        
        # Shortcut branch
        self.conv_shortcut = ConvBNReLU(in_channels, hidden_channels, 1)
        
        # Final conv
        self.conv_out = ConvBNReLU(hidden_channels * 2, out_channels, 1)
    
    def forward(self, x):
        # Main branch
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.conv3(main)
        
        # Shortcut branch
        shortcut = self.conv_shortcut(x)
        
        # Concatenate and output
        out = torch.cat([main, shortcut], dim=1)
        out = self.conv_out(out)
        return out


class CSPNetTiny(nn.Module):
    """
    CSPNet-Tiny Backbone (~6.2M params)
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 2),  # /2
            ConvBNReLU(32, 64, 3, 1),
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2),  # /4
            CSPBlock(128, 128, num_blocks=1),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 2),  # /8
            CSPBlock(256, 256, num_blocks=2),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 512, 3, 2),  # /16
            CSPBlock(512, 512, num_blocks=3),
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNReLU(512, 1024, 3, 2),  # /32
            CSPBlock(1024, 1024, num_blocks=2),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage1(x)      # /4
        c3 = self.stage2(c2)     # /8
        c4 = self.stage3(c3)     # /16
        c5 = self.stage4(c4)    # /32
        
        return [c2, c3, c4, c5]


# ==================== BiFPN-Lite Neck ====================
class BiFPNBlock(nn.Module):
    """BiFPN (Bidirectional Feature Pyramid Network) Lite Block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Top-down path
        self.top_down_conv = ConvBNReLU(channels, channels, 3)
        
        # Bottom-up path
        self.bottom_up_conv = ConvBNReLU(channels, channels, 3)
        
        # Feature fusion weights (learnable)
        self.top_weight = nn.Parameter(torch.ones(2) / 2)
        self.bottom_weight = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, p_in, p_up=None, p_down=None):
        # Top-down fusion
        if p_up is not None:
            p_up_resized = F.interpolate(p_up, size=p_in.shape[2:], mode='nearest')
            weights = F.softmax(self.top_weight, dim=0)
            p_in = weights[0] * p_in + weights[1] * p_up_resized
            p_in = self.top_down_conv(p_in)
        
        # Bottom-up fusion
        if p_down is not None:
            if p_in.shape[2:] != p_down.shape[2:]:
                p_in_resized = F.interpolate(p_in, size=p_down.shape[2:], mode='nearest')
            else:
                p_in_resized = p_in
            weights = F.softmax(self.bottom_weight, dim=0)
            p_out = weights[0] * p_in_resized + weights[1] * p_down
            p_out = self.bottom_up_conv(p_out)
        else:
            p_out = p_in
        
        return p_out


class BiFPNLite(nn.Module):
    """BiFPN-Lite Neck"""
    def __init__(self, in_channels_list=[128, 256, 512, 1024], out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # Channel reduction
        self.reduce_convs = nn.ModuleList([
            ConvBNReLU(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # BiFPN blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels) for _ in range(3)  # 3 layers
        ])
        
        # Output convs
        self.output_convs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, 3) for _ in range(len(in_channels_list))
        ])
    
    def forward(self, features):
        # Reduce channels
        p_features = [conv(feat) for conv, feat in zip(self.reduce_convs, features)]
        
        # BiFPN processing
        for bifpn_block in self.bifpn_blocks:
            # Top-down
            p_features[3] = bifpn_block(p_features[3], p_up=None, p_down=None)
            p_features[2] = bifpn_block(p_features[2], p_up=p_features[3], p_down=None)
            p_features[1] = bifpn_block(p_features[1], p_up=p_features[2], p_down=None)
            p_features[0] = bifpn_block(p_features[0], p_up=p_features[1], p_down=None)
            
            # Bottom-up
            p_features[1] = bifpn_block(p_features[1], p_up=None, p_down=p_features[0])
            p_features[2] = bifpn_block(p_features[2], p_up=None, p_down=p_features[1])
            p_features[3] = bifpn_block(p_features[3], p_up=None, p_down=p_features[2])
        
        # Output
        outputs = [conv(feat) for conv, feat in zip(self.output_convs, p_features)]
        
        return outputs


# ==================== ConvGRU Temporal Module ====================
class ConvGRUCell(nn.Module):
    """ConvGRU Cell"""
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = (kernel_size - 1) // 2
        
        # Reset gate
        self.conv_reset = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        
        # Update gate
        self.conv_update = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        
        # New gate
        self.conv_new = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
    
    def forward(self, x, h=None):
        if h is None:
            b, c, h, w = x.shape
            h = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Reset gate
        reset_gate = torch.sigmoid(self.conv_reset(combined))
        
        # Update gate
        update_gate = torch.sigmoid(self.conv_update(combined))
        
        # New gate
        new_gate = torch.tanh(self.conv_new(torch.cat([x, reset_gate * h], dim=1)))
        
        # Hidden state update
        h_new = (1 - update_gate) * h + update_gate * new_gate
        
        return h_new


class ConvGRU(nn.Module):
    """ConvGRU Temporal Module (học được, không dùng Kalman)"""
    def __init__(self, input_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.gru_cells = nn.ModuleList([
            ConvGRUCell(input_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
    
    def forward(self, x, hidden_states=None):
        """
        Args:
            x: Input features [B, C, H, W]
            hidden_states: List of hidden states from previous frame
        Returns:
            output: Output features [B, C, H, W]
            new_hidden_states: List of new hidden states
        """
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        
        new_hidden_states = []
        current_input = x
        
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(current_input, hidden_states[i])
            new_hidden_states.append(h)
            current_input = h
        
        return current_input, new_hidden_states


# ==================== Depthwise Cross-Correlation ====================
class DepthwiseCrossCorrelation(nn.Module):
    """Depthwise Cross-Correlation for template and search features"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, template, search):
        """
        Args:
            template: Template features [B, C, H_t, W_t]
            search: Search features [B, C, H_s, W_s]
        Returns:
            corr: Cross-correlation features [B, C, H_s, W_s]
        """
        b, c, h_s, w_s = search.shape
        h_t, w_t = template.shape[2:]
        
        # Depthwise cross-correlation
        # Resize template to match search if different sizes
        if h_t != h_s or w_t != w_s:
            template = F.interpolate(template, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Element-wise multiplication (depthwise correlation)
        corr = template * search  # [B, C, H_s, W_s]
        
        # Optional: Add normalization
        # Normalize by L2 norm
        template_norm = F.normalize(template, p=2, dim=1)
        search_norm = F.normalize(search, p=2, dim=1)
        corr = template_norm * search_norm
        
        return corr


# ==================== RPN Head với 9 Anchors ====================
class RPNHead(nn.Module):
    """RPN Head với 9 anchors tối ưu"""
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, 3),
            ConvBNReLU(in_channels, in_channels, 3),
        )
        self.cls_head = nn.Conv2d(in_channels, num_anchors, 1)
        
        # Regression branch
        self.reg_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, 3),
            ConvBNReLU(in_channels, in_channels, 3),
        )
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)  # 4 for bbox (x, y, w, h)
        
        # Initialize
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            cls: Classification logits [B, num_anchors, H, W]
            reg: Regression predictions [B, num_anchors*4, H, W]
        """
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        
        cls = self.cls_head(cls_feat)
        reg = self.reg_head(reg_feat)
        
        return cls, reg


# ==================== Main Model ====================
class TrackingModel(nn.Module):
    """
    Mô hình Object Tracking hoàn chỉnh:
    - Backbone: CSPNet-Tiny
    - Neck: BiFPN-Lite
    - Temporal: ConvGRU
    - Head: Depthwise Cross-Correlation + RPN
    """
    def __init__(self, num_classes=1, num_anchors=9):
        super().__init__()
        
        # Backbone
        self.backbone = CSPNetTiny(in_channels=3)
        
        # Neck
        self.neck = BiFPNLite(in_channels_list=[128, 256, 512, 1024], out_channels=256)
        
        # Temporal (ConvGRU) - áp dụng cho feature level cao nhất
        self.temporal = ConvGRU(input_channels=256, hidden_channels=256, num_layers=2)
        
        # Feature fusion cho tracking
        self.template_conv = ConvBNReLU(256, 256, 3)
        self.search_conv = ConvBNReLU(256, 256, 3)
        
        # Cross-correlation
        self.cross_corr = DepthwiseCrossCorrelation(channels=256)
        
        # RPN Head
        self.rpn_head = RPNHead(in_channels=256, num_anchors=num_anchors)
        
        # Anchor generation (9 anchors tối ưu)
        self.anchor_scales = [0.5, 1.0, 2.0]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = num_anchors
        
        # Hidden state storage
        self.hidden_states = None
    
    def generate_anchors(self, feature_size, stride=16):
        """Generate 9 anchors tối ưu"""
        anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                w = scale * stride * math.sqrt(ratio)
                h = scale * stride / math.sqrt(ratio)
                anchors.append([0, 0, w, h])  # Center format
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, template, search, reset_hidden=False):
        """
        Args:
            template: Template image [B, 3, H, W]
            search: Search image [B, 3, H, W]
            reset_hidden: Reset hidden states
        Returns:
            cls: Classification predictions
            reg: Regression predictions
        """
        if reset_hidden:
            self.hidden_states = None
        
        # Extract features
        template_features = self.backbone(template)
        search_features = self.backbone(search)
        
        # Neck processing
        template_neck = self.neck(template_features)
        search_neck = self.neck(search_features)
        
        # Use highest resolution feature (first one)
        template_feat = template_neck[0]  # [B, 256, H/4, W/4]
        search_feat = search_neck[0]      # [B, 256, H/4, W/4]
        
        # Temporal processing (ConvGRU) on search features
        search_feat, self.hidden_states = self.temporal(search_feat, self.hidden_states)
        
        # Template and search processing
        template_feat = self.template_conv(template_feat)
        search_feat = self.search_conv(search_feat)
        
        # Cross-correlation
        corr_feat = self.cross_corr(template_feat, search_feat)
        
        # RPN Head
        cls, reg = self.rpn_head(corr_feat)
        
        return cls, reg
    
    def reset_hidden_state(self):
        """Reset hidden states"""
        self.hidden_states = None


def count_parameters(model):
    """Đếm số parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = TrackingModel(num_classes=1, num_anchors=9)
    
    # Count parameters
    total_params = count_parameters(model)
    backbone_params = count_parameters(model.backbone)
    
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Backbone parameters: {backbone_params / 1e6:.2f}M")
    
    # Test forward
    template = torch.randn(1, 3, 256, 256)
    search = torch.randn(1, 3, 256, 256)
    
    cls, reg = model(template, search, reset_hidden=True)
    print(f"\nOutput shapes:")
    print(f"  Classification: {cls.shape}")
    print(f"  Regression: {reg.shape}")

