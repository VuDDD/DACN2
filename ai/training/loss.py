import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for binary and multi-channel segmentation.
    Supports smoothing parameter to prevent division by zero and handle sparse masks.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape [B, C, H, W]
            targets: Binary ground truth masks of shape [B, C, H, W] in [0, 1]
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions [B, C, H*W]
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (probs * targets).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets.sum(dim=2)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        # Average across channels and batch
        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Compound Loss function: Weighted combination of BCEWithLogitsLoss and Soft Dice Loss.
    
    BCE provides stable gradient descent across pixel-wise classification,
    while Dice Loss optimizes for spatial overlap and mitigates background-foreground class imbalance.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, class_weights: torch.Tensor = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_fn = DiceLoss()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            # Expand weights to match logits shape [1, C, 1, 1]
            weights = weights.view(1, -1, 1, 1)
            bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        dice = self.dice_fn(logits, targets)
        return (self.bce_weight * bce) + (self.dice_weight * dice)

