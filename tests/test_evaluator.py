import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.evaluation.evaluator import (
    extract_retinal_fov,
    analyze_connected_components,
    evaluate_retinal_findings,
    render_color_mask_overlay,
    calculate_lesion_surface_mm2,
    classify_npdr_severity
)

def test_extract_retinal_fov():
    """Verify FOV circular mask extraction on synthetic fundus image."""
    # Create black image with bright circle in center
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    y, x = np.ogrid[:200, :200]
    dist_from_center = np.sqrt((x - 100)**2 + (y - 100)**2)
    circle_mask = dist_from_center <= 60
    img[circle_mask] = [180, 120, 40] # Retinal color

    fov_mask, retinal_pixels = extract_retinal_fov(img, threshold=15)
    assert retinal_pixels > 0, "Retinal pixels must be positive"
    assert fov_mask.shape == (200, 200), "FOV mask shape must match image"

def test_analyze_connected_components():
    """Verify distinct connected lesion foci counting."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw 3 distinct lesion spots
    mask[10:15, 10:15] = 1 # Spot 1 (25 px)
    mask[30:34, 30:34] = 1 # Spot 2 (16 px)
    mask[70:80, 70:80] = 1 # Spot 3 (100 px)

    count, total_area = analyze_connected_components(mask, min_size=2)
    assert count == 3, f"Expected 3 connected components, got {count}"
    assert total_area == (25 + 16 + 100), f"Expected 141 px, got {total_area}"

def test_render_color_mask_overlay():
    """Verify translucent multi-channel overlay rendering."""
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    masks = np.zeros((5, 100, 100), dtype=np.float32)
    masks[0, 20:40, 20:40] = 1.0 # MA (Red)

    classes = ["MA", "HE", "EX", "SE", "OD"]
    overlay = render_color_mask_overlay(img, masks, classes, alpha=0.5)
    
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    # Center pixel should have higher red channel due to blending
    assert overlay[30, 30, 0] > overlay[10, 10, 0]

def test_analyze_connected_components_micro_noise_filtering():
    """Verify micro-pixel noise (< 3 px) is correctly filtered out."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 2 single-pixel artifacts (area = 1 px each)
    mask[5, 5] = 1
    mask[25, 25] = 1
    # 1 double-pixel artifact (area = 2 px)
    mask[40, 40] = 1
    mask[40, 41] = 1
    # 1 true lesion spot (area = 16 px)
    mask[60:64, 60:64] = 1

    # With min_size = 3: the 1px and 2px artifacts should be eliminated, keeping only the 16px lesion
    count, total_area = analyze_connected_components(mask, min_size=3)
    assert count == 1, f"Expected 1 valid lesion, got {count}"
    assert total_area == 16, f"Expected 16 px, got {total_area}"

def test_evaluate_retinal_findings_mm2_conversion():
    """Verify conversion from pixel area to physical mm² based on FOV mask."""
    # Synthetic FOV: 10,000 pixels
    fov_mask = np.zeros((100, 100), dtype=bool)
    fov_mask[0:100, 0:100] = True # 10,000 px

    # Synthetic lesion: 100 pixels of MA
    pred_masks = np.zeros((5, 100, 100), dtype=np.float32)
    pred_masks[0, 10:20, 10:20] = 1.0 # 100 px of MA
    classes = ["MA", "HE", "EX", "SE", "OD"]

    res = evaluate_retinal_findings(
        pred_masks=pred_masks,
        class_names=classes,
        fov_mask=fov_mask,
        standard_fov_mm2=1090.0
    )

    # 100 / 10,000 = 1.0% -> mm² = 0.01 * 1090.0 = 10.9 mm²
    ma_item = next(item for item in res["lesion_breakdown"] if item["lesion_type"] == "MA")
    assert np.isclose(ma_item["area_mm2"], 10.9, atol=1e-2), f"Expected 10.9 mm², got {ma_item.get('area_mm2')}"
    assert np.isclose(res["total_lesion_mm2"], 10.9, atol=1e-2)
    assert "npdr_assessment" in res
    assert res["npdr_assessment"]["grade"] in ["Mild", "Moderate", "Severe", "No DR"]

def test_classify_npdr_severity():
    """Verify ICDR severity grading logic for Mild, Moderate, Severe NPDR and No DR."""
    # 1. No DR: no lesions detected
    breakdown_clean = [
        {"lesion_type": "MA", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
        {"lesion_type": "HE", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
        {"lesion_type": "EX", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
        {"lesion_type": "SE", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
    ]
    res_clean = classify_npdr_severity(breakdown_clean, total_lesion_percentage=0.0)
    assert res_clean["grade"] == "No DR"

    # 2. Mild NPDR: only microaneurysms (MA > 0)
    breakdown_mild = [
        {"lesion_type": "MA", "count": 4, "area_pixels": 30, "area_percentage": 0.03},
        {"lesion_type": "HE", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
        {"lesion_type": "EX", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
        {"lesion_type": "SE", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
    ]
    res_mild = classify_npdr_severity(breakdown_mild, total_lesion_percentage=0.03)
    assert res_mild["grade"] == "Mild"

    # 3. Moderate NPDR: MA + Hard Exudates (EX) or mild HE
    breakdown_mod = [
        {"lesion_type": "MA", "count": 6, "area_pixels": 50, "area_percentage": 0.05},
        {"lesion_type": "HE", "count": 3, "area_pixels": 80, "area_percentage": 0.08},
        {"lesion_type": "EX", "count": 4, "area_pixels": 120, "area_percentage": 0.12},
        {"lesion_type": "SE", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
    ]
    res_mod = classify_npdr_severity(breakdown_mod, total_lesion_percentage=0.25)
    assert res_mod["grade"] == "Moderate"

    # 4. Severe NPDR: extensive HE or high lesion percentage (> 1.5% FOV)
    breakdown_sev = [
        {"lesion_type": "MA", "count": 15, "area_pixels": 200, "area_percentage": 0.2},
        {"lesion_type": "HE", "count": 20, "area_pixels": 1800, "area_percentage": 1.8},
        {"lesion_type": "EX", "count": 10, "area_pixels": 500, "area_percentage": 0.5},
        {"lesion_type": "SE", "count": 6, "area_pixels": 300, "area_percentage": 0.3},
    ]
    res_sev = classify_npdr_severity(breakdown_sev, total_lesion_percentage=2.8)
    assert res_sev["grade"] == "Severe"


