"""
Hàm Loss Multi-Head cho bài toán Nhận diện thuộc tính người đi bộ (PAR) UPAR (40 attributes).
Sử dụng Weighted CrossEntropyLoss cho Age Head (multi-class) và Weighted BCEWithLogitsLoss cho các Head còn lại
nhằm giải quyết triệt để vấn đề mất cân bằng nhãn (Class Imbalance).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union
from datasets.upar.loader import build_batch_multi_head_targets, HEAD_SPECS

DEFAULT_LOSS_WEIGHTS = {
    "age": 1.0,
    "gender": 1.0,
    "hair": 1.0,
    "upper_length": 1.0,
    "upper_color": 1.0,
    "lower_length": 1.0,
    "lower_color": 1.0,
    "lower_type": 1.0,
    "bag": 1.0,
    "glasses": 1.0,
    "hat": 1.0
}


class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss for Multi-label Binary Classification with optional positive weighting.
    FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super(BinaryFocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pw)
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1.0 - probs)
        focal_factor = (1.0 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        return loss.mean()


class MultiHeadPARLoss(nn.Module):
    """
    Multi-Head Loss for Pedestrian Attribute Recognition with Class Imbalance Weighting.
    Supports optional Focal Loss for high-imbalance multi-label heads (e.g., upper_color, lower_color).
    """
    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
        pos_weights_dict: Dict[str, torch.Tensor] = None,
        age_class_weights: torch.Tensor = None,
        use_focal_heads: list = None,
        focal_gamma: float = 2.0
    ):
        super(MultiHeadPARLoss, self).__init__()
        self.weights = DEFAULT_LOSS_WEIGHTS.copy()
        if loss_weights is not None:
            self.weights.update(loss_weights)
            
        self.pos_weights_dict = pos_weights_dict or {}
        self.age_class_weights = age_class_weights
        self.use_focal_heads = use_focal_heads or []
        self.focal_gamma = focal_gamma
        
        if self.age_class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=self.age_class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            
        self.bce_heads = {}
        for head_name in DEFAULT_LOSS_WEIGHTS.keys():
            if head_name == "age":
                continue
            pw = self.pos_weights_dict.get(head_name, None)
            if head_name in self.use_focal_heads:
                self.bce_heads[head_name] = BinaryFocalLossWithLogits(gamma=self.focal_gamma, pos_weight=pw)
            elif pw is not None:
                self.bce_heads[head_name] = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self.bce_heads[head_name] = nn.BCEWithLogitsLoss()

    def compute_pos_weights_from_labels(self, raw_labels: np.ndarray, max_w: float = 50.0):
        """
        Calculate positive class weights (pos_weight) for each head from dataset labels using sqrt formula.
        """
        if isinstance(raw_labels, torch.Tensor):
            raw_labels = raw_labels.cpu().numpy()

        pos_counts = np.sum(raw_labels == 1, axis=0)
        neg_counts = np.sum(raw_labels == 0, axis=0)
        ratio = np.clip(np.sqrt(neg_counts / (pos_counts + 1e-5)), 0.5, max_w)
        
        # Age class weights (inverse frequency)
        age_counts = pos_counts[:3]
        age_w = len(raw_labels) / (3.0 * (age_counts + 1e-5))
        age_w = age_w / np.mean(age_w)
        self.age_class_weights = torch.tensor(age_w, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.age_class_weights)

        # Per-head BCE pos_weights
        print("\n" + "=" * 60)
        print(f"CALCULATED POS_WEIGHTS PER HEAD (max_w={max_w}, formula=sqrt(neg/pos))")
        print("=" * 60)
        print(f"  - {'age':<15}: Age CE Weights = {np.round(self.age_class_weights.numpy(), 3)}")

        for head_name, spec in HEAD_SPECS.items():
            if head_name == "age":
                continue
            indices = spec["indices"]
            pw_arr = ratio[indices]
            pw = torch.tensor(pw_arr, dtype=torch.float32)
            if head_name in self.use_focal_heads:
                self.bce_heads[head_name] = BinaryFocalLossWithLogits(gamma=self.focal_gamma, pos_weight=pw)
            else:
                self.bce_heads[head_name] = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"  - {head_name:<15}: pos_weight min={pw_arr.min():.2f}, max={pw_arr.max():.2f}, mean={pw_arr.mean():.2f} -> {np.round(pw_arr, 2)}")
            
        print("=" * 60 + "\n", flush=True)

    def to(self, device):
        super(MultiHeadPARLoss, self).to(device)
        if self.age_class_weights is not None:
            self.age_class_weights = self.age_class_weights.to(device)
            self.ce_loss = nn.CrossEntropyLoss(weight=self.age_class_weights)
            
        for head_name in self.bce_heads:
            if head_name in self.pos_weights_dict:
                pw = self.pos_weights_dict[head_name].to(device)
                self.pos_weights_dict[head_name] = pw
                if head_name in self.use_focal_heads:
                    self.bce_heads[head_name] = BinaryFocalLossWithLogits(gamma=self.focal_gamma, pos_weight=pw)
                else:
                    self.bce_heads[head_name] = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self.bce_heads[head_name] = self.bce_heads[head_name].to(device)
        return self

    def compute_losses(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and per-head loss dictionary.
        """
        if isinstance(targets, torch.Tensor):
            targets_dict = build_batch_multi_head_targets(targets)
        else:
            targets_dict = targets

        head_losses = {}
        device = logits_dict["age"].device
        
        # 1. Age Head (Weighted CrossEntropy Loss)
        ce_loss_fn = self.ce_loss.to(device) if self.ce_loss is not None else nn.CrossEntropyLoss().to(device)
        head_losses["age"] = ce_loss_fn(logits_dict["age"], targets_dict["age"])
        
        # 2. Other Heads (Weighted BCE Loss)
        for head_name in DEFAULT_LOSS_WEIGHTS.keys():
            if head_name == "age":
                continue
            if head_name in self.bce_heads:
                bce_fn = self.bce_heads[head_name].to(device)
            else:
                bce_fn = nn.BCEWithLogitsLoss().to(device)
            head_losses[head_name] = bce_fn(logits_dict[head_name], targets_dict[head_name])

        # Total Weighted Loss
        total_loss = torch.tensor(0.0, device=device)
        for head_name, loss_val in head_losses.items():
            w = self.weights.get(head_name, 1.0)
            total_loss = total_loss + w * loss_val

        return total_loss, head_losses

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        total_loss, _ = self.compute_losses(logits_dict, targets)
        return total_loss


# Class aliases for backward compatibility
MaskedBCEWithLogitsLoss = MultiHeadPARLoss
WeightedBCEWithLogitsLoss = MultiHeadPARLoss
