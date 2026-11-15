import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ai.config import CLASS_COLORS

def extract_retinal_fov(image_rgb: np.ndarray, threshold: int = 15) -> Tuple[np.ndarray, int]:
    """
    Extracts the valid Field of View (FOV) circular mask of the retina from an RGB fundus photo.
    Background outside the fundus circle is excluded to ensure accurate percentage quantification.
    
    Returns:
        (fov_mask [H, W] boolean, total_retinal_pixels int)
    """
    # Green channel typically has the highest contrast for retinal fundus imaging
    if image_rgb.ndim == 3:
        green = image_rgb[:, :, 1]
    else:
        green = image_rgb

    fov_mask = (green > threshold)
    
    # Simple morphological fill if scipy/cv2 available, or basic array logic
    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fov_mask = cv2.morphologyEx(fov_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        fov_mask = (fov_mask > 0)
    except Exception:
        pass

    total_retinal_pixels = int(np.sum(fov_mask))
    if total_retinal_pixels == 0:
        total_retinal_pixels = fov_mask.shape[0] * fov_mask.shape[1] # fallback to full image
        fov_mask = np.ones(fov_mask.shape, dtype=bool)

    return fov_mask, total_retinal_pixels


def analyze_connected_components(binary_mask: np.ndarray, min_size: int = 2) -> Tuple[int, int]:
    """
    Counts distinct lesion foci using Connected Component Analysis.
    
    Args:
        binary_mask: 2D binary numpy array [H, W]
        min_size: Minimum pixel threshold to filter single-pixel acquisition artifacts.
    Returns:
        (foci_count, total_lesion_pixels)
    """
    b = (binary_mask > 0.5).astype(np.uint8)
    total_pixels = int(np.sum(b))
    if total_pixels == 0:
        return 0, 0

    try:
        import cv2
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b, connectivity=8)
        # Label 0 is background; evaluate labels 1 to num_labels-1
        valid_count = 0
        filtered_pixels = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                valid_count += 1
                filtered_pixels += area
        return valid_count, (filtered_pixels if filtered_pixels > 0 else total_pixels)
    except Exception:
        # Fallback using scipy ndimage if cv2 is unavailable
        try:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(b)
            return int(num_features), total_pixels
        except Exception:
            return 1 if total_pixels > 0 else 0, total_pixels


def evaluate_retinal_findings(
    pred_masks: np.ndarray,
    class_names: List[str],
    image_rgb: Optional[np.ndarray] = None,
    fov_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Performs clinical medical evaluation:
    - Extracts or validates FOV mask
    - Counts connected lesion foci per class
    - Calculates pixel area and proportion relative to valid retinal surface area
    
    Args:
        pred_masks: [C, H, W] in range [0, 1]
        class_names: list of class names (e.g. ['MA', 'HE', 'EX', 'SE', 'OD'])
        image_rgb: optional [H, W, 3] image for FOV extraction
        fov_mask: optional precomputed boolean mask [H, W]
    """
    H, W = pred_masks.shape[1], pred_masks.shape[2]

    # Establish FOV
    if fov_mask is not None:
        valid_fov = (fov_mask > 0.5)
        retinal_pixels = int(np.sum(valid_fov))
    elif image_rgb is not None:
        valid_fov, retinal_pixels = extract_retinal_fov(image_rgb)
    else:
        valid_fov = np.ones((H, W), dtype=bool)
        retinal_pixels = H * W

    lesion_breakdown = []
    total_abnormal_pixels = 0
    aggregate_mask = np.zeros((H, W), dtype=bool)

    for i, c_name in enumerate(class_names):
        if i >= pred_masks.shape[0]:
            break
        
        c_mask = (pred_masks[i] > 0.5) & valid_fov
        foci_cnt, area_px = analyze_connected_components(c_mask)
        pct_fov = (area_px / max(1, retinal_pixels)) * 100.0

        lesion_breakdown.append({
            "lesion_type": c_name,
            "count": foci_cnt,
            "area_pixels": area_px,
            "area_percentage": round(float(pct_fov), 4)
        })

        # OD (Optic Disc) is an anatomical landmark, not a pathological lesion
        if c_name != "OD":
            total_abnormal_pixels += area_px
            aggregate_mask = aggregate_mask | c_mask

    total_abnormal_pct = (np.sum(aggregate_mask) / max(1, retinal_pixels)) * 100.0

    return {
        "retinal_fov_pixels": retinal_pixels,
        "total_lesion_pixels": int(np.sum(aggregate_mask)),
        "total_lesion_percentage": round(float(total_abnormal_pct), 4),
        "lesion_breakdown": lesion_breakdown,
        "fov_mask": valid_fov
    }


def render_color_mask_overlay(
    image_rgb: np.ndarray,
    pred_masks: np.ndarray,
    class_names: List[str],
    alpha: float = 0.45
) -> np.ndarray:
    """
    Renders multi-layer translucent lesion boundaries atop the original retinal image.
    Color Palette:
      MA: Red (239, 68, 68)
      HE: Orange (249, 115, 22)
      EX: Yellow (234, 179, 8)
      SE: Green (34, 197, 94)
      OD: Blue (59, 130, 246)
      VESSEL: Purple (168, 85, 247)
    """
    H, W = image_rgb.shape[0], image_rgb.shape[1]
    overlay = image_rgb.copy().astype(np.float32)

    # Blend each class where mask > 0.5
    for i, c_name in enumerate(class_names):
        if i >= pred_masks.shape[0]:
            break
        color = np.array(CLASS_COLORS.get(c_name, (255, 255, 255)), dtype=np.float32)
        mask = (pred_masks[i] > 0.5)
        if np.any(mask):
            # Alpha blend in place for active pixels
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    return np.clip(overlay, 0, 255).astype(np.uint8)

