import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ai.config import CLASS_COLORS

# Standard adult human retina visible Field of View area in 45°-50° fundus cameras (~1090 mm²)
STANDARD_RETINAL_FOV_MM2: float = 1090.0

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
    
    # Morphological fill
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


def analyze_connected_components(binary_mask: np.ndarray, min_size: int = 3) -> Tuple[int, int]:
    """
    Counts distinct lesion foci using Connected Component Analysis.
    Filters micro-pixel noise / acquisition artifacts below min_size.
    
    Args:
        binary_mask: 2D binary numpy array [H, W]
        min_size: Minimum pixel threshold to filter single-pixel or micro artifacts (default: 3 px).
    Returns:
        (foci_count, total_filtered_lesion_pixels)
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
        return valid_count, filtered_pixels
    except Exception:
        # Fallback using scipy ndimage if cv2 is unavailable
        try:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(b)
            if num_features == 0:
                return 0, 0
            sizes = ndimage.sum(b, labeled_array, range(1, num_features + 1))
            valid_sizes = [s for s in sizes if s >= min_size]
            return len(valid_sizes), int(sum(valid_sizes))
        except Exception:
            return (1 if total_pixels >= min_size else 0), (total_pixels if total_pixels >= min_size else 0)


def calculate_lesion_surface_mm2(
    area_pixels: int,
    retinal_fov_pixels: int,
    standard_fov_mm2: float = STANDARD_RETINAL_FOV_MM2
) -> float:
    """
    Converts lesion pixel surface into calibrated physical metric area in square millimeters (mm²).
    Formula: (area_pixels / retinal_fov_pixels) * standard_fov_mm2
    """
    if retinal_fov_pixels <= 0:
        return 0.0
    return float((area_pixels / retinal_fov_pixels) * standard_fov_mm2)


def classify_npdr_severity(
    lesion_breakdown: List[Dict[str, Any]],
    total_lesion_percentage: float
) -> Dict[str, Any]:
    """
    Clinical grading of Diabetic Retinopathy severity based on the International Clinical
    Diabetic Retinopathy (ICDR) severity scale:
    - No DR: No microaneurysms or retinal hemorrhages/exudates detected.
    - Mild NPDR: Microaneurysms (MA) only, without significant hemorrhages or exudates.
    - Moderate NPDR: More than just microaneurysms (MA accompanied by moderate HE, EX, or SE).
    - Severe NPDR: Extensive retinal hemorrhages (HE >= 15 foci or HE area >= 0.5% FOV),
                   or significant soft exudates/cotton-wool spots (SE >= 5 foci),
                   or overall pathological lesion load >= 1.5% FOV.
    """
    counts = {item["lesion_type"]: item.get("count", 0) for item in lesion_breakdown}
    pcts = {item["lesion_type"]: item.get("area_percentage", 0.0) for item in lesion_breakdown}

    ma_count = counts.get("MA", 0)
    he_count = counts.get("HE", 0)
    ex_count = counts.get("EX", 0)
    se_count = counts.get("SE", 0)

    he_pct = pcts.get("HE", 0.0)

    # 1. No DR Check
    if total_lesion_percentage == 0.0 and ma_count == 0 and he_count == 0 and ex_count == 0 and se_count == 0:
        return {
            "grade": "No DR",
            "icdr_level": 0,
            "description": "Không phát hiện dấu hiệu tổn thương vi mạch hoặc bệnh võng mạc đái tháo đường.",
            "recommendation": "Khám sàng lọc định kỳ đáy mắt hàng năm (12 tháng/lần)."
        }

    # 2. Severe NPDR Check
    if he_count >= 15 or he_pct >= 0.5 or se_count >= 5 or total_lesion_percentage >= 1.5:
        return {
            "grade": "Severe",
            "icdr_level": 3,
            "description": "Bệnh võng mạc đái tháo đường không tăng sinh giai đoạn nặng (Severe NPDR). Xuất huyết diện rộng hoặc nhiều ổ xuất tiết mềm (cotton-wool spots) chỉ điểm thiếu máu cục bộ võng mạc.",
            "recommendation": "Khẩn cấp: Chuyển khám chuyên khoa mắt trong vòng 2-4 tuần để chụp mạch huỳnh quang (FFA) và cân nhắc quang đông laser võng mạc hoặc tiêm nội nhãn Anti-VEGF."
        }

    # 3. Moderate NPDR Check
    if ma_count > 0 and (he_count > 0 or ex_count > 0 or se_count > 0 or total_lesion_percentage > 0.05):
        return {
            "grade": "Moderate",
            "icdr_level": 2,
            "description": "Bệnh võng mạc đái tháo đường không tăng sinh mức độ trung bình (Moderate NPDR). Xuất hiện vi phình mạch kèm xuất huyết rải rác hoặc xuất tiết cứng / lipid lắng đọng.",
            "recommendation": "Cần khám chuyên khoa mắt trong vòng 2-3 tháng, kiểm soát chặt chẽ đường huyết (HbA1c < 7.0%) và huyết áp."
        }

    # 4. Mild NPDR (Microaneurysms only)
    return {
        "grade": "Mild",
        "icdr_level": 1,
        "description": "Bệnh võng mạc đái tháo đường không tăng sinh giai đoạn sớm/nhẹ (Mild NPDR). Chỉ phát hiện các vi phình mạch (Microaneurysms) đơn độc.",
        "recommendation": "Theo dõi tái khám đáy mắt sau 6-12 tháng, duy trì ổn định đường huyết và kiểm soát yếu tố nguy cơ tim mạch."
    }


def evaluate_retinal_findings(
    pred_masks: np.ndarray,
    class_names: List[str],
    image_rgb: Optional[np.ndarray] = None,
    fov_mask: Optional[np.ndarray] = None,
    standard_fov_mm2: float = STANDARD_RETINAL_FOV_MM2
) -> Dict[str, Any]:
    """
    Performs clinical medical evaluation:
    - Extracts or validates FOV mask
    - Counts connected lesion foci per class with micro-artifact noise filtering
    - Calculates pixel area, physical metric mm², and proportion relative to valid retinal surface area
    - Determines clinical NPDR severity stage according to ICDR criteria
    
    Args:
        pred_masks: [C, H, W] in range [0, 1]
        class_names: list of class names (e.g. ['MA', 'HE', 'EX', 'SE', 'OD'])
        image_rgb: optional [H, W, 3] image for FOV extraction
        fov_mask: optional precomputed boolean mask [H, W]
        standard_fov_mm2: physical calibrated area of retinal FOV in mm² (default: 1090.0)
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
        foci_cnt, area_px = analyze_connected_components(c_mask, min_size=3)
        pct_fov = (area_px / max(1, retinal_pixels)) * 100.0
        area_mm2 = calculate_lesion_surface_mm2(area_px, retinal_pixels, standard_fov_mm2)

        lesion_breakdown.append({
            "lesion_type": c_name,
            "count": int(foci_cnt),
            "area_pixels": int(area_px),
            "area_mm2": round(float(area_mm2), 4),
            "area_percentage": round(float(pct_fov), 4)
        })

        # OD (Optic Disc) and VESSEL are anatomical structures, not pathological lesions
        if c_name not in ("OD", "VESSEL"):
            total_abnormal_pixels += area_px
            aggregate_mask = aggregate_mask | (c_mask if area_px > 0 else np.zeros((H, W), dtype=bool))

    total_abnormal_px = int(np.sum(aggregate_mask))
    total_abnormal_pct = (total_abnormal_px / max(1, retinal_pixels)) * 100.0
    total_abnormal_mm2 = calculate_lesion_surface_mm2(total_abnormal_px, retinal_pixels, standard_fov_mm2)

    npdr_assessment = classify_npdr_severity(lesion_breakdown, total_abnormal_pct)

    return {
        "retinal_fov_pixels": int(retinal_pixels),
        "total_lesion_pixels": total_abnormal_px,
        "total_lesion_mm2": round(float(total_abnormal_mm2), 4),
        "total_lesion_percentage": round(float(total_abnormal_pct), 4),
        "lesion_breakdown": lesion_breakdown,
        "npdr_assessment": npdr_assessment,
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
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    return np.clip(overlay, 0, 255).astype(np.uint8)
