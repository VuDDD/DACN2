import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ai_core.datasets.medical_dataset import MedicalLesionDataset
from ai_core.utils.augmentations import get_train_transforms, get_val_transforms
from ai_core.models.unet import UNet
def dice_coef(preds, targets, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return((2.0*intersection + smooth) / (union + smooth)).mean().item()
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, preds, targets, smooth = 1e-6):
        bce_loss = self.bce(preds, targets)
        probs = torch.sigmoid(preds)
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()
        return 0.5 * bce_loss + 0.5 * dice_loss
def train_model(epochs=30, batch_size=4, lr=1e-4, img_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang huấn luyện trên thiết bị: {device}")

    train_ds = MedicalLesionDataset("ai_core/data/processed/train.csv", transform=get_train_transforms(img_size))
    val_ds = MedicalLesionDataset("ai_core/data/processed/val.csv", transform=get_val_transforms(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs("ai_core/weights", exist_ok=True)
    best_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                val_dice += dice_coef(outputs, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "ai_core/weights/best_model.pth")
            print(f"  --> Đã lưu best_model.pth (Val Dice: {best_dice:.4f})")

    # Vẽ biểu đồ Loss và Dice Score
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_dice"], color="green", label="Val Dice Score")
    plt.title("Validation Dice Score")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("ai_core/weights/training_curves.png", dpi=200)
    print("Hoàn tất huấn luyện! Đã xuất biểu đồ tại ai_core/weights/training_curves.png")

if __name__ == "__main__":
    train_model(epochs=30, batch_size=4, lr=1e-4)