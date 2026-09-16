"""
Mô hình mạng nơ-ron Multi-Task / Multi-Head classification PAR cho bộ dữ liệu UPAR (40 Attributes).
Bao gồm Shared Backbone (ResNet50), Spatial Attention, Shared Feature Representation (512-dim),
và 11 Task-Specific Classification Heads cho các thuộc tính người đi bộ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from models.hydraplus.backbone import build_backbone


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_map = self.conv(x)  # (B, 1, H, W)
        return x * attn_map + x   # Residual attention feature map


class UnifiedPARModel(nn.Module):
    def __init__(self, num_attributes: int = 40, backbone_name: str = 'resnet50', pretrained: bool = True, dropout: float = 0.4):
        super(UnifiedPARModel, self).__init__()
        self.num_attributes = num_attributes
        self.backbone_name = backbone_name
        
        self.backbone, self.feature_dim = build_backbone(name=backbone_name, pretrained=pretrained)
        self.attention = SpatialAttention(self.feature_dim)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared Projection Representation Layer
        self.shared_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        # 11 Task-Specific Classification Heads
        self.heads = nn.ModuleDict({
            "age": nn.Linear(512, 3),            # Age-Young, Adult, Old
            "gender": nn.Linear(512, 1),         # Gender-Female
            "hair": nn.Linear(512, 3),           # Hair-Short, Long, Bald
            "upper_length": nn.Linear(512, 1),   # Upper-Length-Short
            "upper_color": nn.Linear(512, 12),   # Upper 12 Colors
            "lower_length": nn.Linear(512, 1),   # Lower-Length-Short
            "lower_color": nn.Linear(512, 12),   # Lower 12 Colors
            "lower_type": nn.Linear(512, 2),     # Lower Trousers/Shorts, Skirt/Dress
            "bag": nn.Linear(512, 2),            # Backpack, Bag
            "glasses": nn.Linear(512, 2),        # Glasses-Normal, Sun
            "hat": nn.Linear(512, 1)             # Accessory-Hat
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        attn_features = self.attention(features)
        pooled = self.pooling(attn_features)
        flattened = torch.flatten(pooled, 1)
        shared_repr = self.shared_proj(flattened)
        
        outputs = {}
        for head_name, head_layer in self.heads.items():
            outputs[head_name] = head_layer(shared_repr)
            
        return outputs

    def predict_40_logits(self, head_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate head logits back into a unified 40-dim raw logits tensor.
        Order matches 40 UPAR attributes (00..39).
        """
        return torch.cat([
            head_outputs["age"],          # 0..2 (3)
            head_outputs["gender"],       # 3 (1)
            head_outputs["hair"],         # 4..6 (3)
            head_outputs["upper_length"], # 7 (1)
            head_outputs["upper_color"],  # 8..19 (12)
            head_outputs["lower_length"], # 20 (1)
            head_outputs["lower_color"],  # 21..32 (12)
            head_outputs["lower_type"],   # 33..34 (2)
            head_outputs["bag"],          # 35..36 (2)
            head_outputs["glasses"],      # 37..38 (2)
            head_outputs["hat"]           # 39 (1)
        ], dim=1)

    def predict_40_probabilities(self, head_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert head outputs to activation probabilities.
        For Age multi-class: Uses argmax one-hot / scaled probabilities so winning class >= 0.5.
        For other multi-label heads: Uses Sigmoid.
        """
        softmax_age = F.softmax(head_outputs["age"], dim=1)
        max_idx = torch.argmax(softmax_age, dim=1, keepdim=True)
        # Argmax one-hot for Age multi-class attributes
        prob_age = torch.zeros_like(softmax_age).scatter_(1, max_idx, 1.0)
        
        prob_gender = torch.sigmoid(head_outputs["gender"])
        prob_hair = torch.sigmoid(head_outputs["hair"])
        prob_upper_len = torch.sigmoid(head_outputs["upper_length"])
        prob_upper_col = torch.sigmoid(head_outputs["upper_color"])
        prob_lower_len = torch.sigmoid(head_outputs["lower_length"])
        prob_lower_col = torch.sigmoid(head_outputs["lower_color"])
        prob_lower_type = torch.sigmoid(head_outputs["lower_type"])
        prob_bag = torch.sigmoid(head_outputs["bag"])
        prob_glasses = torch.sigmoid(head_outputs["glasses"])
        prob_hat = torch.sigmoid(head_outputs["hat"])
        
        return torch.cat([
            prob_age, prob_gender, prob_hair, prob_upper_len,
            prob_upper_col, prob_lower_len, prob_lower_col,
            prob_lower_type, prob_bag, prob_glasses, prob_hat
        ], dim=1)
