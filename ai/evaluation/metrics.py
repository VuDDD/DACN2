import numpy as np
import torch
from typing import Dict, Any, Union

def compute_binary_metrics(pred_binary: np.ndarray, target_binary: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    """
    Computes standard biomedical segmentation metrics on flattened 1D binary masks.
    pred_binary: 0 or 1 binary array
    target_binary: 0 or 1 binary array
    """
    pred_b = (pred_binary > 0.5).astype(np.uint8)
    targ_b = (target_binary > 0.5).astype(np.uint8)

    tp = np.sum((pred_b == 1) & (targ_b == 1))
    fp = np.sum((pred_b == 1) & (targ_b == 0))
    fn = np.sum((pred_b == 0) & (targ_b == 1))
    tn = np.sum((pred_b == 0) & (targ_b == 0))

    # Dice Coefficient
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    
    # Intersection over Union (Jaccard)
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    # Precision (Positive Predictive Value)
    precision = (tp + eps) / (tp + fp + eps)
    
    # Recall (Sensitivity / True Positive Rate)
    recall = (tp + eps) / (tp + fn + eps)
    
    # Specificity (True Negative Rate)
    specificity = (tn + eps) / (tn + fp + eps)
    
    # F1-Score
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def calculate_metrics_multiclass(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: list,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes per-class and macro-averaged metrics for multi-channel predictions.
    Shapes: [C, H, W] or [B, C, H, W]
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # If shape is [B, C, H, W], flatten batch and spatial per channel
    if preds.ndim == 4:
        # B, C, H, W
        num_classes = preds.shape[1]
    elif preds.ndim == 3:
        num_classes = preds.shape[0]
    else:
        num_classes = 1

    per_class = {}
    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    specificity_list = []

    for c in range(min(num_classes, len(class_names))):
        c_name = class_names[c]
        if preds.ndim == 4:
            p_c = preds[:, c, :, :].flatten()
            t_c = targets[:, c, :, :].flatten()
        elif preds.ndim == 3:
            p_c = preds[c, :, :].flatten()
            t_c = targets[c, :, :].flatten()
        else:
            p_c = preds.flatten()
            t_c = targets.flatten()

        c_metrics = compute_binary_metrics(p_c, t_c)
        per_class[c_name] = c_metrics
        
        # Collect for macro-averaging
        dice_list.append(c_metrics["dice"])
        iou_list.append(c_metrics["iou"])
        precision_list.append(c_metrics["precision"])
        recall_list.append(c_metrics["recall"])
        specificity_list.append(c_metrics["specificity"])

    return {
        "macro_avg": {
            "dice": float(np.mean(dice_list)) if dice_list else 0.0,
            "iou": float(np.mean(iou_list)) if iou_list else 0.0,
            "precision": float(np.mean(precision_list)) if precision_list else 0.0,
            "recall": float(np.mean(recall_list)) if recall_list else 0.0,
            "specificity": float(np.mean(specificity_list)) if specificity_list else 0.0,
        },
        "per_class": per_class
    }
