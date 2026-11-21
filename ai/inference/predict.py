import os
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List

import torch
from ai.config import (
    DEVICE, IMAGE_SIZE, IDRID_CLASSES, DRIVE_CLASSES,
    CHECKPOINT_DIR, CLASS_COLORS
)
from ai.models.unet import UNet
from ai.models.attention_unet import AttentionUNet
from ai.evaluation.evaluator import evaluate_retinal_findings, render_color_mask_overlay, extract_retinal_fov
from ai.evaluation.metrics import compute_binary_metrics, calculate_metrics_multiclass

class RetinalPredictor:
    """
    Production-grade AI inference engine for:
    1. Multi-lesion retinal segmentation (IDRiD: MA, HE, EX, SE, OD)
    2. Retinal micro-vascular tree extraction (DRIVE: Vessels within FOV)
    """
    def __init__(
        self,
        task_type: str = "IDRID_LESIONS",
        model_name: str = "UNet_Baseline",
        checkpoint_dir: str = CHECKPOINT_DIR,
        device: Optional[torch.device] = None
    ):
        self.task_type = task_type
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device or DEVICE

        if task_type == "DRIVE_VESSELS":
            self.classes = DRIVE_CLASSES
            self.n_classes = len(DRIVE_CLASSES)
            self.prefix = "drive"
        else:
            self.classes = IDRID_CLASSES
            self.n_classes = len(IDRID_CLASSES)
            self.prefix = "idrid"

        self.loaded_checkpoint_path: Optional[str] = None
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Instantiates model architecture and attempts to load trained checkpoint weights."""
        is_att = "attention" in self.model_name.lower()
        if is_att:
            model = AttentionUNet(n_channels=3, n_classes=self.n_classes)
            tag = "attention_unet"
        else:
            model = UNet(n_channels=3, n_classes=self.n_classes)
            tag = "unet"

        model = model.to(self.device)
        model.eval()

        # Prioritize designated production checkpoint weights
        candidates = [
            os.path.join(self.checkpoint_dir, f"{self.prefix}_{tag}_best.pth"),
            os.path.join(self.checkpoint_dir, f"{self.prefix}_{tag}_best_model.pth"),
            os.path.join(self.checkpoint_dir, f"{self.prefix}_best.pth"),
            os.path.join(self.checkpoint_dir, f"{self.prefix}_best_model.pth"),
            os.path.join(self.checkpoint_dir, "best_model.pth"),
        ]

        for ckpt_path in candidates:
            if os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    model.load_state_dict(state_dict)
                    self.loaded_checkpoint_path = ckpt_path
                    print(f"[RetinalPredictor] Successfully loaded checkpoint weights from {ckpt_path}")
                    break
                except Exception as e:
                    print(f"[RetinalPredictor] Warning: Error loading {ckpt_path} ({e}). Trying next candidate...")

        if not self.loaded_checkpoint_path:
            print(f"[RetinalPredictor] Note: Running baseline initialization (no checkpoint found for {self.prefix}_{tag}).")

        return model

    def preprocess_image(self, img_pil: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocesses PIL Image: resize, green-channel CLAHE, normalize to tensor."""
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize(IMAGE_SIZE, Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32)

        # Apply CLAHE to green channel
        try:
            import cv2
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            green = img_np[:, :, 1].astype(np.uint8)
            img_np[:, :, 1] = clahe.apply(green)
        except Exception:
            pass

        norm_arr = img_np / 255.0
        tensor = torch.from_numpy(np.transpose(norm_arr, (2, 0, 1))).unsqueeze(0).float()
        return tensor.to(self.device), np.array(img_resized)

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        ground_truth_path: Optional[str] = None,
        fov_mask_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes full inference and medical quantification pipeline.
        Returns:
            Dictionary with masks, overlay image, lesion statistics, and validation metrics.
        """
        t0 = time.time()
        
        with Image.open(image_path) as raw_img:
            tensor_in, rgb_img_resized = self.preprocess_image(raw_img)

        # Forward pass
        logits = self.model(tensor_in)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() # [C, H, W]
        binary_preds = (probs > 0.5).astype(np.float32)

        # Load FOV mask if provided, else extract automatically from image
        fov_mask = None
        if fov_mask_path and os.path.exists(fov_mask_path):
            try:
                with Image.open(fov_mask_path) as fov_im:
                    fov_im = fov_im.convert("L").resize(IMAGE_SIZE, Image.NEAREST)
                    fov_mask = (np.array(fov_im) > 0)
            except Exception:
                pass

        if fov_mask is None:
            fov_mask, _ = extract_retinal_fov(rgb_img_resized)

        # Medical quantification
        eval_result = evaluate_retinal_findings(
            pred_masks=binary_preds,
            class_names=self.classes,
            image_rgb=rgb_img_resized,
            fov_mask=fov_mask
        )

        # Render color overlay
        overlay_rgb = render_color_mask_overlay(
            image_rgb=rgb_img_resized,
            pred_masks=binary_preds,
            class_names=self.classes,
            alpha=0.45
        )

        elapsed_ms = int((time.time() - t0) * 1000)

        # Evaluate against Ground Truth if provided
        metrics_dict = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            try:
                with Image.open(ground_truth_path) as gt_im:
                    gt_im = gt_im.convert("L").resize(IMAGE_SIZE, Image.NEAREST)
                    gt_arr = (np.array(gt_im) > 0).astype(np.float32)
                
                # Compare primary lesion or vessel channel
                m_res = compute_binary_metrics(binary_preds[0].flatten(), gt_arr.flatten())
                metrics_dict = m_res
            except Exception as e:
                print(f"[RetinalPredictor] Error evaluating ground truth ({e})")

        return {
            "task_type": self.task_type,
            "model_name": self.model_name,
            "classes": self.classes,
            "raw_probabilities": probs,
            "binary_masks": binary_preds,
            "overlay_image": overlay_rgb,
            "fov_mask": fov_mask,
            "quantification": eval_result,
            "metrics": metrics_dict,
            "loaded_checkpoint_path": self.loaded_checkpoint_path,
            "processing_time_ms": elapsed_ms
        }

