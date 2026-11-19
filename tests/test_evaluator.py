import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.evaluation.evaluator import (
    extract_retinal_fov,
    analyze_connected_components,
    evaluate_retinal_findings,
    render_color_mask_overlay
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

