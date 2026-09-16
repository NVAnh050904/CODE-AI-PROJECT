"""
Các chỉ số đánh giá cho mô hình Multi-Head Pedestrian Attribute Recognition (PAR) UPAR (40 Attributes).
Tính toán chỉ số theo từng Head (Age, Gender, Hair, Upper/Lower Color/Length, Accessories)
và chỉ số tổng thể cho 40 thuộc tính (mA, Precision, Recall, F1-score).
"""
import torch
import numpy as np
from typing import Dict, Any
from datasets.upar.loader import UPAR_ATTRIBUTES, HEAD_SPECS


def compute_par_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """
    Compute 40-attribute PAR metrics (mA, Precision, Recall, F1) from (N, 40) probabilities and targets.
    """
    N, num_attrs = probs.shape
    preds = (probs >= 0.5).astype(int)

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    per_attribute_metrics = {}

    for i in range(num_attrs):
        attr_name = UPAR_ATTRIBUTES[i] if i < len(UPAR_ATTRIBUTES) else f"attr_{i}"
        
        valid_mask = (targets[:, i] != 2)  # Mask out missing/unknown if any
        valid_preds = preds[:, i][valid_mask]
        valid_targets = targets[:, i][valid_mask]

        pos_gt = np.sum(valid_targets == 1)
        neg_gt = np.sum(valid_targets == 0)

        tp = np.sum((valid_preds == 1) & (valid_targets == 1))
        tn = np.sum((valid_preds == 0) & (valid_targets == 0))
        fp = np.sum((valid_preds == 1) & (valid_targets == 0))
        fn = np.sum((valid_preds == 0) & (valid_targets == 1))

        pos_acc = tp / (pos_gt + 1e-5) if pos_gt > 0 else 1.0
        neg_acc = tn / (neg_gt + 1e-5) if neg_gt > 0 else 1.0
        attr_ma = 0.5 * (pos_acc + neg_acc)

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)

        acc_list.append(attr_ma)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        per_attribute_metrics[attr_name] = {
            "mA": float(attr_ma),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    overall_mA = float(np.mean(acc_list))
    overall_precision = float(np.mean(precision_list))
    overall_recall = float(np.mean(recall_list))
    overall_f1 = float(np.mean(f1_list))

    return {
        "mA": overall_mA,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "per_attribute": per_attribute_metrics
    }


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Average Precision (AP) for a single binary classification attribute.
    """
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = (y_true_sorted == 1).astype(int)
    fp = (y_true_sorted == 0).astype(int)
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    ap = np.sum(tp * precisions) / (np.sum(tp) + 1e-8) if np.sum(tp) > 0 else 0.0
    return float(ap)


def compute_head_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute task-specific evaluation metrics for each of the 11 classification heads.
    Includes Average Precision (mAP) for multi-label color heads.
    """
    head_metrics = {}
    
    for head_name, spec in HEAD_SPECS.items():
        indices = spec["indices"]
        head_probs = probs[:, indices]
        head_targets = targets[:, indices]
        
        if spec["type"] == "multiclass":
            # Multi-class Age Head
            pred_classes = np.argmax(head_probs, axis=1)
            target_classes = np.argmax(head_targets, axis=1)
            acc = float(np.mean(pred_classes == target_classes))
            head_metrics[head_name] = {"accuracy": acc, "f1": acc}
        else:
            # Binary & Multi-label Heads
            pred_binary = (head_probs >= 0.5).astype(int)
            acc = float(np.mean(pred_binary == head_targets))
            
            tp = np.sum((pred_binary == 1) & (head_targets == 1))
            fp = np.sum((pred_binary == 1) & (head_targets == 0))
            fn = np.sum((pred_binary == 0) & (head_targets == 1))
            
            precision = float(tp / (tp + fp + 1e-5))
            recall = float(tp / (tp + fn + 1e-5))
            f1 = float(2 * precision * recall / (precision + recall + 1e-5))
            
            # Compute Mean Average Precision (mAP) for color heads
            aps = [compute_ap(head_targets[:, c], head_probs[:, c]) for c in range(head_probs.shape[1])]
            map_score = float(np.mean(aps))
            
            head_metrics[head_name] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mAP": map_score
            }
            
    return head_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Any = None
) -> Dict[str, Any]:
    """
    Evaluate Multi-Head UnifiedPARModel on a dataloader.
    
    Returns metrics dict containing total loss, head losses, head metrics, and 40-attribute metrics.
    """
    model.eval()
    all_probs = []
    all_targets = []
    val_loss = 0.0
    val_acc = 0.0
    head_loss_accum = {}
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            raw_targets = batch[1]
            targets_device = raw_targets.to(device)
            
            logits_dict = model(imgs)
            probs = model.predict_40_probabilities(logits_dict)
            
            if criterion is not None:
                if hasattr(criterion, 'compute_losses'):
                    total_l, head_l = criterion.compute_losses(logits_dict, targets_device)
                    val_loss += total_l.item()
                    for h_name, h_val in head_l.items():
                        head_loss_accum[h_name] = head_loss_accum.get(h_name, 0.0) + h_val.item()
                else:
                    l = criterion(logits_dict, targets_device)
                    val_loss += l.item()
                    
            preds = (probs > 0.5).float()
            batch_acc = (preds == targets_device).float().mean().item()
            val_acc += batch_acc
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(raw_targets.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    print("\n" + "=" * 60)
    print("EVALUATION DEBUG: PRED_POS_RATE vs TRUE_POS_RATE PER HEAD")
    print("=" * 60)
    for head_name, spec in HEAD_SPECS.items():
        indices = spec["indices"]
        h_probs = all_probs[:, indices]
        h_targets = all_targets[:, indices]
        
        pred_pos_rate = np.round(np.mean(h_probs >= 0.5, axis=0), 4).tolist()
        true_pos_rate = np.round(np.mean(h_targets == 1, axis=0), 4).tolist()
        print(f"  {head_name:<15}: pred_pos_rate={pred_pos_rate}, true_pos_rate={true_pos_rate}")
    print("=" * 60)

    # Color Heads Threshold Optimization Search
    print("COLOR HEADS THRESHOLD OPTIMIZATION (F1-score)")
    print("=" * 60)
    for color_head in ["upper_color", "lower_color"]:
        indices = HEAD_SPECS[color_head]["indices"]
        h_probs = all_probs[:, indices]
        h_targets = all_targets[:, indices]
        
        default_f1s = []
        opt_f1s = []
        best_ts = []
        for c in range(h_probs.shape[1]):
            y_true = h_targets[:, c]
            y_score = h_probs[:, c]
            
            p_def = (y_score >= 0.5).astype(int)
            tp_def = np.sum((p_def == 1) & (y_true == 1))
            fp_def = np.sum((p_def == 1) & (y_true == 0))
            fn_def = np.sum((p_def == 0) & (y_true == 1))
            f1_def = 2 * tp_def / (2 * tp_def + fp_def + fn_def + 1e-8)
            default_f1s.append(f1_def)
            
            best_t = 0.5
            best_f1 = f1_def
            for t in np.linspace(0.01, 0.99, 99):
                p_t = (y_score >= t).astype(int)
                tp = np.sum((p_t == 1) & (y_true == 1))
                fp = np.sum((p_t == 1) & (y_true == 0))
                fn = np.sum((p_t == 0) & (y_true == 1))
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            opt_f1s.append(best_f1)
            best_ts.append(round(float(best_t), 2))
            
        print(f"  {color_head:<15}: Default F1@0.5 = {np.mean(default_f1s):.4f} -> Opt F1 = {np.mean(opt_f1s):.4f}")
        print(f"    Best Thresholds per channel: {best_ts}")
    print("=" * 60 + "\n", flush=True)

    metrics = compute_par_metrics(all_probs, all_targets)
    metrics["head_metrics"] = compute_head_metrics(all_probs, all_targets)
    metrics["loss"] = val_loss / total_batches if total_batches > 0 else 0.0
    metrics["accuracy"] = val_acc / total_batches if total_batches > 0 else 0.0
    
    if head_loss_accum and total_batches > 0:
        metrics["head_losses"] = {h: v / total_batches for h, v in head_loss_accum.items()}

    return metrics
