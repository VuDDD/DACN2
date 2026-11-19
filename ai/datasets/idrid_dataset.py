import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional, Any

# Fallback-tolerant torch import while wheel is finishing installation
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass

from ai.config import IDRID_DIR, IDRID_CLASSES, IMAGE_SIZE

class IDRiDDataset(Dataset):
    """
    IDRiD (Indian Diabetic Retinopathy Image Dataset) Segmentation Dataset.
    Loads RGB retinal fundus images and synchronizes 5 lesion masks:
      1. MA: Microaneurysms
      2. HE: Haemorrhages
      3. EX: Hard Exudates
      4. SE: Soft Exudates
      5. OD: Optic Disc
    
    If a particular lesion mask is absent for a given eye, an all-zero binary mask is produced.
    """
    def __init__(
        self,
        root_dir: str = IDRID_DIR,
        split: str = "train", # 'train', 'val', 'test'
        val_ratio: float = 0.15,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        apply_clahe: bool = True,
        seed: int = 42
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.classes = IDRID_CLASSES

        self.samples = self._load_file_index(split, val_ratio, seed)

    def _load_file_index(self, split: str, val_ratio: float, seed: int) -> List[Dict[str, str]]:
        samples = []
        
        # IDRiD has official directories: 'a. Training Set' and 'b. Testing Set'
        orig_base = os.path.join(self.root_dir, "1. Original Images")
        gt_base = os.path.join(self.root_dir, "2. All Segmentation Groundtruths")

        if split == "test":
            img_dir = os.path.join(orig_base, "b. Testing Set")
            gt_dir = os.path.join(gt_base, "b. Testing Set")
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        else:
            img_dir = os.path.join(orig_base, "a. Training Set")
            gt_dir = os.path.join(gt_base, "a. Training Set")
            all_train_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            
            # Deterministic train/validation split without data leakage
            rng = np.random.RandomState(seed)
            indices = np.arange(len(all_train_paths))
            rng.shuffle(indices)
            n_val = int(len(all_train_paths) * val_ratio)
            
            val_indices = set(indices[:n_val])
            if split == "val":
                img_paths = [all_train_paths[i] for i in range(len(all_train_paths)) if i in val_indices]
            else: # 'train'
                img_paths = [all_train_paths[i] for i in range(len(all_train_paths)) if i not in val_indices]

        # Map each original image to groundtruth masks for all 5 lesions
        subfolder_map = {
            "MA": "1. Microaneurysms",
            "HE": "2. Haemorrhages",
            "EX": "3. Hard Exudates",
            "SE": "4. Soft Exudates",
            "OD": "5. Optic Disc"
        }

        for p in img_paths:
            base_name = os.path.splitext(os.path.basename(p))[0] # e.g. 'IDRiD_01'
            masks_map = {}
            for c in self.classes:
                subf = subfolder_map[c]
                # Pattern: IDRiD_01_MA.tif
                cand1 = os.path.join(gt_dir, subf, f"{base_name}_{c}.tif")
                cand2 = os.path.join(gt_dir, subf, f"{base_name}_{c}.TIF")
                if os.path.exists(cand1):
                    masks_map[c] = cand1
                elif os.path.exists(cand2):
                    masks_map[c] = cand2
                else:
                    masks_map[c] = None # No lesion present in this image

            samples.append({
                "image_id": base_name,
                "image_path": p,
                "masks": masks_map
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Dict[str, Any]]:
        sample = self.samples[idx]
        
        # Load Original RGB Image
        with Image.open(sample["image_path"]) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize(self.target_size, Image.BILINEAR)
            img_arr = np.array(img, dtype=np.float32)

        # CLAHE Contrast Enhancement on Green channel
        if self.apply_clahe:
            try:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                green = img_arr[:, :, 1].astype(np.uint8)
                img_arr[:, :, 1] = clahe.apply(green)
            except Exception:
                pass

        # Normalize to [0, 1]
        img_arr = img_arr / 255.0
        # Transpose to [C, H, W]
        img_tensor = np.transpose(img_arr, (2, 0, 1))

        # Load 5-channel Groundtruth Mask
        mask_channels = []
        for c in self.classes:
            m_path = sample["masks"][c]
            if m_path and os.path.exists(m_path):
                try:
                    with Image.open(m_path) as m_img:
                        m_img = m_img.resize(self.target_size, Image.NEAREST)
                        m_arr = np.array(m_img, dtype=np.float32)
                        # Binarize
                        m_arr = (m_arr > 0).astype(np.float32)
                except Exception:
                    m_arr = np.zeros(self.target_size, dtype=np.float32)
            else:
                m_arr = np.zeros(self.target_size, dtype=np.float32)
            
            mask_channels.append(m_arr)

        # Stack into [5, H, W]
        masks_tensor = np.stack(mask_channels, axis=0)

        # Convert to torch Tensor if torch available
        try:
            import torch
            img_tensor = torch.from_numpy(img_tensor).float()
            masks_tensor = torch.from_numpy(masks_tensor).float()
        except Exception:
            pass

        meta = {
            "image_id": sample["image_id"],
            "orig_size": (orig_w, orig_h),
            "split": self.split
        }

        return img_tensor, masks_tensor, meta

def export_idrid_statistics(root_dir: str = IDRID_DIR, output_csv: str = "data/idrid_statistics.csv") -> pd.DataFrame:
    """
    Computes rigorous statistical distribution of IDRiD lesions and masks,
    saving the audit results to a CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    stats = []

    for split in ["train", "test"]:
        sub = "a. Training Set" if split == "train" else "b. Testing Set"
        orig_dir = os.path.join(root_dir, "1. Original Images", sub)
        gt_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", sub)

        if not os.path.exists(orig_dir):
            continue

        images = sorted(glob.glob(os.path.join(orig_dir, "*.jpg")))
        for img_p in images:
            b_name = os.path.splitext(os.path.basename(img_p))[0]
            row = {"image_id": b_name, "split": split}
            
            subfolders = {
                "MA": "1. Microaneurysms",
                "HE": "2. Haemorrhages",
                "EX": "3. Hard Exudates",
                "SE": "4. Soft Exudates",
                "OD": "5. Optic Disc"
            }
            
            total_lesion_px = 0
            for c, s_name in subfolders.items():
                m_p = os.path.join(gt_dir, s_name, f"{b_name}_{c}.tif")
                if not os.path.exists(m_p):
                    m_p = os.path.join(gt_dir, s_name, f"{b_name}_{c}.TIF")
                
                if os.path.exists(m_p):
                    try:
                        with Image.open(m_p) as m_im:
                            m_arr = np.array(m_im)
                            px_cnt = int(np.sum(m_arr > 0))
                            row[f"{c}_present"] = 1
                            row[f"{c}_pixels"] = px_cnt
                            if c != "OD":
                                total_lesion_px += px_cnt
                    except Exception:
                        row[f"{c}_present"] = 0
                        row[f"{c}_pixels"] = 0
                else:
                    row[f"{c}_present"] = 0
                    row[f"{c}_pixels"] = 0

            row["total_pathological_pixels"] = total_lesion_px
            stats.append(row)

    df = pd.DataFrame(stats)
    df.to_csv(output_csv, index=False)
    return df
