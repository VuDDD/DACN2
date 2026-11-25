import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ai.config import (
    DEVICE, IMAGE_SIZE, IDRID_CLASSES, DRIVE_CLASSES,
    CHECKPOINT_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)
from ai.models.unet import UNet
from ai.models.attention_unet import AttentionUNet
from ai.training.loss import BCEDiceLoss, ComboFocalDiceLoss
from ai.training.augmentations import RetinalAugmentor
from ai.evaluation.metrics import calculate_metrics_multiclass, compute_binary_metrics


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, augmentor=None, is_drive=False):
    model.train()
    running_loss = 0.0
    use_amp = (device.type == "cuda")

    for batch in loader:
        if is_drive:
            images, targets, fov, _ = batch
        else:
            images, targets, _ = batch

        # Apply augmentation if provided
        if augmentor and np.random.rand() < 0.7:
            img_np = images.numpy()
            targ_np = targets.numpy()
            for b in range(img_np.shape[0]):
                img_np[b], targ_np[b] = augmentor(img_np[b], targ_np[b])
            images = torch.from_numpy(img_np).float()
            targets = torch.from_numpy(targ_np).float()

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, class_names, is_drive=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        if is_drive:
            images, targets, fov, _ = batch
        else:
            images, targets, _ = batch

        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu())
        all_targets.append(targets.cpu())

    epoch_loss = running_loss / len(loader.dataset)

    cat_preds = torch.cat(all_preds, dim=0).numpy()
    cat_targets = torch.cat(all_targets, dim=0).numpy()

    if is_drive:
        metrics = compute_binary_metrics(cat_preds[:, 0, :, :].flatten(), cat_targets[:, 0, :, :].flatten())
        val_dice = metrics["dice"]
        val_iou = metrics["iou"]
    else:
        res = calculate_metrics_multiclass(cat_preds, cat_targets, class_names=class_names)
        val_dice = res["macro_avg"]["dice"]
        val_iou = res["macro_avg"]["iou"]

    return epoch_loss, val_dice, val_iou


def plot_and_save_curves(history_df: pd.DataFrame, output_dir: str, prefix: str, loss_label: str = "Loss"):
    """Generates and saves visual training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = history_df["epoch"]

        # Loss Curve
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["train_loss"], label="Train Loss", color="#0284c7", lw=2)
        plt.plot(epochs, history_df["val_loss"], label="Val Loss", color="#f97316", lw=2)
        plt.title(f"{prefix} - Training and Validation Loss Curve", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel(loss_label)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_loss_curve.png"), dpi=200)
        plt.close()

        # Dice Score Curve
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["val_dice"], label="Validation Dice", color="#10b981", lw=2)
        plt.title(f"{prefix} - Validation Dice Coefficient", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_dice_curve.png"), dpi=200)
        plt.close()

        # IoU Curve
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["val_iou"], label="Validation IoU", color="#8b5cf6", lw=2)
        plt.title(f"{prefix} - Validation IoU (Jaccard)", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("IoU Score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_iou_curve.png"), dpi=200)
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to render matplotlib plots ({e})")


def run_training(
    dataset_name: str = "idrid",
    model_name: str = "unet",
    loss_name: str = "auto",
    epochs: int = 15,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device_str: str = "auto",
    output_dir: str = CHECKPOINT_DIR
):
    os.makedirs(output_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    is_drive = (dataset_name.lower() == "drive")
    if is_drive:
        from ai.datasets.drive_dataset import DRIVEDataset
        class_names = DRIVE_CLASSES
        n_classes = len(DRIVE_CLASSES)
        full_drive_ds = DRIVEDataset(split="train")
        n_total = len(full_drive_ds)
        n_val = max(1, int(round(0.2 * n_total)))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_drive_ds, [n_train, n_val], generator=generator)
    else:
        from ai.datasets.idrid_dataset import IDRiDDataset
        class_names = IDRID_CLASSES
        n_classes = len(IDRID_CLASSES)
        train_ds = IDRiDDataset(split="train")
        val_ds = IDRiDDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Architecture
    if model_name.lower() in ("attention_unet", "attention-unet", "att_unet"):
        model = AttentionUNet(n_channels=3, n_classes=n_classes)
        model_tag = "attention_unet"
    else:
        model = UNet(n_channels=3, n_classes=n_classes)
        model_tag = "unet"

    model = model.to(device)

    # Resolve Loss Function
    if loss_name == "auto":
        resolved_loss = "focal_dice" if (model_tag == "attention_unet" and not is_drive) else "bce_dice"
    else:
        resolved_loss = loss_name.lower()

    if resolved_loss == "focal_dice":
        criterion = ComboFocalDiceLoss(focal_weight=0.5, dice_weight=0.5, alpha=0.75, gamma=2.0)
        loss_label = "Focal + Dice Loss"
    else:
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        loss_label = "BCE + Dice Loss"

    print("=== Starting Training Session ===")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_tag} | Loss: {loss_label} | ImageSize: {IMAGE_SIZE} | Device: {device} | Epochs: {epochs} | Batch: {batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    augmentor = RetinalAugmentor()

    prefix = f"{dataset_name.lower()}_{model_tag}"
    best_dice = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, augmentor, is_drive)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device, class_names, is_drive)
        elapsed = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dice)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | LR: {curr_lr:.2e}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_dice": round(val_dice, 5),
            "val_iou": round(val_iou, 5),
            "lr": curr_lr
        })

        # Save best model checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(output_dir, f"{prefix}_best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_dice,
                "dataset": dataset_name,
                "model_name": model_tag,
                "loss_name": resolved_loss,
                "image_size": IMAGE_SIZE,
                "n_classes": n_classes,
                "class_names": class_names
            }, best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (Dice: {best_dice:.4f})")

    # Save last model checkpoint
    last_path = os.path.join(output_dir, f"{prefix}_last_model.pth")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "dataset": dataset_name,
        "model_name": model_tag,
        "loss_name": resolved_loss,
        "image_size": IMAGE_SIZE,
        "n_classes": n_classes,
        "class_names": class_names
    }, last_path)

    # Save history CSV and plots
    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, f"{prefix}_training_history.csv")
    df_hist.to_csv(csv_path, index=False)
    plot_and_save_curves(df_hist, output_dir, prefix, loss_label=loss_label)

    print("=== Training Completed ===")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"History saved to {csv_path}")
    return df_hist, best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net / Attention U-Net on IDRiD or DRIVE")
    parser.add_argument("--dataset", type=str, default="idrid", choices=["idrid", "drive"])
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "attention_unet"])
    parser.add_argument("--loss", type=str, default="auto", choices=["auto", "bce_dice", "focal_dice"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_training(
        dataset_name=args.dataset,
        model_name=args.model,
        loss_name=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device
    )