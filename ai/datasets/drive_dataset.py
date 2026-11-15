import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional, Any

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass

from ai.config import DRIVE_DIR, IMAGE_SIZE

class DRIVEDataset(Dataset):
    """
    DRIVE (Digital Retinal Images for Vessel Extraction) Dataset.
    Dedicated pipeline for binary retinal blood vessel segmentation within valid Field of View (FOV).
    
    Training: 20 images with vessel Ground Truth (1st_manual) and FOV mask.
    Testing: 20 images with FOV mask. (Ground truth not fabricated for hidden test benchmark).
    """
    def __init__(
        self,
        root_dir: str = DRIVE_DIR,
        split: str = "train", # 'train' or 'test'
        target_size: Tuple[int, int] = IMAGE_SIZE,
        apply_clahe: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.apply_clahe = apply_clahe

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, str]]:
        samples = []
        split_folder = "training" if self.split == "train" else "test"
        base = os.path.join(self.root_dir, split_folder)

        img_dir = os.path.join(base, "images")
        mask_dir = os.path.join(base, "mask")
        gt_dir = os.path.join(base, "1st_manual") if self.split == "train" else None

        if not os.path.exists(img_dir):
            return samples

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".jpg", ".png"))])

        for f in img_files:
            # Pattern: 21_training.tif -> prefix '21'
            prefix = f.split("_")[0]
            img_path = os.path.join(img_dir, f)

            # Find matching FOV mask
            # e.g., 21_training_mask.gif or 01_test_mask.gif
            fov_cand = glob.glob(os.path.join(mask_dir, f"{prefix}_*_mask.*"))
            fov_path = fov_cand[0] if fov_cand else None

            # Find matching ground truth vessel mask (train split only)
            gt_path = None
            if gt_dir and os.path.exists(gt_dir):
                gt_cand = glob.glob(os.path.join(gt_dir, f"{prefix}_manual1.*"))
                if gt_cand:
                    gt_path = gt_cand[0]

            samples.append({
                "image_id": prefix,
                "image_path": img_path,
                "gt_path": gt_path,
                "fov_path": fov_path
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        sample = self.samples[idx]

        # Load RGB image
        with Image.open(sample["image_path"]) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize(self.target_size, Image.BILINEAR)
            img_arr = np.array(img, dtype=np.float32)

        # Contrast enhancement on green channel (vessels stand out darkest on green)
        if self.apply_clahe:
            try:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                green = img_arr[:, :, 1].astype(np.uint8)
                img_arr[:, :, 1] = clahe.apply(green)
            except Exception:
                pass

        img_arr = img_arr / 255.0
        img_tensor = np.transpose(img_arr, (2, 0, 1)) # [3, H, W]

        # Load Ground Truth vessel mask
        if sample["gt_path"] and os.path.exists(sample["gt_path"]):
            with Image.open(sample["gt_path"]) as gt_img:
                gt_img = gt_img.convert("L").resize(self.target_size, Image.NEAREST)
                gt_arr = (np.array(gt_img, dtype=np.float32) > 0).astype(np.float32)
        else:
            gt_arr = np.zeros(self.target_size, dtype=np.float32)
        
        gt_tensor = np.expand_dims(gt_arr, axis=0) # [1, H, W]

        # Load FOV mask
        if sample["fov_path"] and os.path.exists(sample["fov_path"]):
            with Image.open(sample["fov_path"]) as fov_img:
                fov_img = fov_img.convert("L").resize(self.target_size, Image.NEAREST)
                fov_arr = (np.array(fov_img, dtype=np.float32) > 0).astype(np.float32)
        else:
            fov_arr = np.ones(self.target_size, dtype=np.float32)

        fov_tensor = np.expand_dims(fov_arr, axis=0) # [1, H, W]

        try:
            import torch
            img_tensor = torch.from_numpy(img_tensor).float()
            gt_tensor = torch.from_numpy(gt_tensor).float()
            fov_tensor = torch.from_numpy(fov_tensor).float()
        except Exception:
            pass

        meta = {
            "image_id": sample["image_id"],
            "has_gt": sample["gt_path"] is not None,
            "orig_size": (orig_w, orig_h),
            "split": self.split
        }

        return img_tensor, gt_tensor, fov_tensor, meta


def export_drive_statistics(root_dir: str = DRIVE_DIR, output_csv: str = "data/drive_statistics.csv") -> pd.DataFrame:
    """Computes vessel pixel counts and FOV statistics on the DRIVE dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    stats = []

    ds = DRIVEDataset(root_dir=root_dir, split="train")
    for i in range(len(ds)):
        _, gt, fov, meta = ds[i]
        gt_np = gt.numpy() if hasattr(gt, "numpy") else gt
        fov_np = fov.numpy() if hasattr(fov, "numpy") else fov

        vessel_px = int(np.sum(gt_np > 0.5))
        fov_px = int(np.sum(fov_np > 0.5))
        density = (vessel_px / max(1, fov_px)) * 100.0

        stats.append({
            "image_id": meta["image_id"],
            "split": "train",
            "vessel_pixels": vessel_px,
            "fov_pixels": fov_px,
            "vessel_density_pct": round(density, 3)
        })

    # Test images (without GT)
    ds_test = DRIVEDataset(root_dir=root_dir, split="test")
    for i in range(len(ds_test)):
        _, _, fov, meta = ds_test[i]
        fov_np = fov.numpy() if hasattr(fov, "numpy") else fov
        fov_px = int(np.sum(fov_np > 0.5))
        stats.append({
            "image_id": meta["image_id"],
            "split": "test",
            "vessel_pixels": "N/A (Benchmark Hidden)",
            "fov_pixels": fov_px,
            "vessel_density_pct": "N/A"
        })

    df = pd.DataFrame(stats)
    df.to_csv(output_csv, index=False)
    return df
