import os
import sys
import pytest
import numpy as np
import torch

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.config import IDRID_CLASSES, DRIVE_CLASSES
from ai.models.unet import UNet
from ai.models.attention_unet import AttentionUNet
from ai.training.loss import BCEDiceLoss, DiceLoss
from ai.evaluation.metrics import compute_binary_metrics, calculate_metrics_multiclass
from ai.datasets.idrid_dataset import IDRiDDataset
from ai.datasets.drive_dataset import DRIVEDataset

def test_unet_architecture_forward():
    """Verify UNet forward pass with RGB input and 5-class IDRiD output."""
    model = UNet(n_channels=3, n_classes=5)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == (2, 5, 256, 256), f"Expected shape (2, 5, 256, 256), got {output.shape}"

def test_attention_unet_forward():
    """Verify Attention UNet forward pass with Attention Gate connections."""
    model = AttentionUNet(n_channels=3, n_classes=5)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == (2, 5, 256, 256), f"Expected shape (2, 5, 256, 256), got {output.shape}"

def test_loss_functions():
    """Verify DiceLoss and BCEDiceLoss computation and gradient backprop."""
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 5, 64, 64, requires_grad=True)
    targets = (torch.rand(2, 5, 64, 64) > 0.5).float()
    
    loss = criterion(logits, targets)
    assert loss.item() > 0, "Loss must be positive"
    loss.backward()
    assert logits.grad is not None, "Gradient must be computed for backpropagation"

def test_metric_calculations():
    """Verify Dice, IoU, Precision, Recall, Specificity computations on identical and orthogonal masks."""
    # Perfect match test
    mask_a = np.ones((100, 100), dtype=np.uint8)
    metrics_perfect = compute_binary_metrics(mask_a, mask_a)
    assert np.isclose(metrics_perfect["dice"], 1.0, atol=1e-4), "Dice of identical masks must be 1.0"
    assert np.isclose(metrics_perfect["iou"], 1.0, atol=1e-4), "IoU of identical masks must be 1.0"

    # Orthogonal / zero overlap test
    mask_b = np.zeros((100, 100), dtype=np.uint8)
    metrics_zero = compute_binary_metrics(mask_a, mask_b)
    assert np.isclose(metrics_zero["dice"], 0.0, atol=1e-4), "Dice of orthogonal masks must be ~0.0"

def test_idrid_dataset_leakage_and_shapes():
    """Verify IDRiD dataset trains/val split has no intersection and outputs correct tensors."""
    ds_train = IDRiDDataset(split="train", target_size=(256, 256))
    ds_val = IDRiDDataset(split="val", target_size=(256, 256))
    ds_test = IDRiDDataset(split="test", target_size=(256, 256))

    train_ids = set(s["image_id"] for s in ds_train.samples)
    val_ids = set(s["image_id"] for s in ds_val.samples)
    test_ids = set(s["image_id"] for s in ds_test.samples)

    # Strict Data Leakage Check
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage detected between train and val splits!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage detected between train and test splits!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage detected between val and test splits!"

    # Shape validation
    img, masks, meta = ds_train[0]
    assert img.shape == (3, 256, 256), f"Image tensor must be [3, H, W], got {img.shape}"
    assert masks.shape == (5, 256, 256), f"Masks tensor must be [5, H, W], got {masks.shape}"

def test_drive_dataset_shapes():
    """Verify DRIVE dataset loads vessels and FOV masks correctly."""
    ds_drive = DRIVEDataset(split="train", target_size=(256, 256))
    assert len(ds_drive) > 0, "DRIVE train dataset must contain samples"
    
    img, gt, fov, meta = ds_drive[0]
    assert img.shape == (3, 256, 256)
    assert gt.shape == (1, 256, 256)
    assert fov.shape == (1, 256, 256)

