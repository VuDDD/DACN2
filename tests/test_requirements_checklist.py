"""
Acceptance Test Suite: Verification of Member 1 (Vu) & Member 2 (Hanh) Tasks (2.2 & 2.3).
Run with:
    python -m pytest tests/test_requirements_checklist.py -v
"""
import csv
import os
import re
import sys
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BACKEND_DIR)

from ai.config import CHECKPOINT_DIR, DRIVE_CLASSES, IDRID_CLASSES, IMAGE_SIZE
from ai.datasets.drive_dataset import DRIVEDataset
from ai.datasets.idrid_dataset import IDRiDDataset
from ai.evaluation.evaluator import (
    STANDARD_RETINAL_FOV_MM2,
    analyze_connected_components,
    classify_npdr_severity,
    evaluate_retinal_findings,
    extract_retinal_fov,
    render_color_mask_overlay,
)
from ai.inference.predict import RetinalPredictor
from ai.training.loss import ComboFocalDiceLoss, FocalLoss


# ==============================================================================
# 2.2. MEMBER 1 — VU (Backend, Evaluator & Integration)
# ==============================================================================


def test_task_1_1_connected_components_and_noise_filter():
    """Task 1.1: Connected Components (8-connectivity) + micro-noise filtering (< 3 px)."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[5, 5] = 1  # 1 px noise
    mask[10, 10:12] = 1  # 2 px noise
    mask[30:35, 30:35] = 1  # 25 px valid lesion
    mask[80:84, 80:85] = 1  # 20 px valid lesion

    count, total_area = analyze_connected_components(mask, min_size=3)
    assert count == 2, f"Expected 2 valid lesions after noise filtering, got {count}"
    assert total_area == 45, f"Expected 45 px total area, got {total_area}"


def test_task_1_1_fov_mm2_conversion_and_npdr_grading():
    """Task 1.1: Pixel-to-mm² conversion on FOV mask & Mild/Moderate/Severe NPDR staging."""
    fov_mask = np.ones((100, 100), dtype=bool)  # 10,000 FOV pixels
    pred_masks = np.zeros((5, 100, 100), dtype=np.float32)
    pred_masks[0, 10:12, 10:12] = 1.0  # MA: 4 px -> 0.04% FOV (<= 0.05% -> Mild)

    res = evaluate_retinal_findings(
        pred_masks=pred_masks,
        class_names=IDRID_CLASSES,
        fov_mask=fov_mask,
        standard_fov_mm2=STANDARD_RETINAL_FOV_MM2,
    )
    ma_item = next(x for x in res["lesion_breakdown"] if x["lesion_type"] == "MA")
    expected_mm2 = (4 / 10000) * STANDARD_RETINAL_FOV_MM2
    assert np.isclose(ma_item["area_mm2"], expected_mm2, atol=1e-2)
    assert res["npdr_assessment"]["grade"] == "Mild"

    # Moderate NPDR
    mod_res = classify_npdr_severity(
        [
            {"lesion_type": "MA", "count": 4, "area_pixels": 40, "area_percentage": 0.04},
            {"lesion_type": "HE", "count": 2, "area_pixels": 60, "area_percentage": 0.06},
            {"lesion_type": "EX", "count": 3, "area_pixels": 100, "area_percentage": 0.10},
            {"lesion_type": "SE", "count": 0, "area_pixels": 0, "area_percentage": 0.0},
        ],
        total_lesion_percentage=0.20,
    )
    assert mod_res["grade"] == "Moderate"

    # Severe NPDR
    sev_res = classify_npdr_severity(
        [
            {"lesion_type": "MA", "count": 10, "area_pixels": 120, "area_percentage": 0.12},
            {"lesion_type": "HE", "count": 18, "area_pixels": 900, "area_percentage": 0.90},
            {"lesion_type": "EX", "count": 8, "area_pixels": 500, "area_percentage": 0.50},
            {"lesion_type": "SE", "count": 5, "area_pixels": 250, "area_percentage": 0.25},
        ],
        total_lesion_percentage=1.77,
    )
    assert sev_res["grade"] == "Severe"


def test_task_1_2_checkpoint_autoload_and_pipeline():
    """Task 1.2: inference_pipeline & RetinalPredictor auto-load idrid_attention_unet_best.pth and drive_unet_best.pth."""
    pred_idrid = RetinalPredictor(task_type="IDRID_LESIONS", model_name="Attention_UNet")
    assert pred_idrid.loaded_checkpoint_path is not None
    assert os.path.basename(pred_idrid.loaded_checkpoint_path) == "idrid_attention_unet_best.pth"

    pred_drive = RetinalPredictor(task_type="DRIVE_VESSELS", model_name="UNet_Baseline")
    assert pred_drive.loaded_checkpoint_path is not None
    assert os.path.basename(pred_drive.loaded_checkpoint_path) == "drive_unet_best.pth"

    pipeline_path = os.path.join(BACKEND_DIR, "app", "services", "inference_pipeline.py")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "RetinalPredictor" in src
    assert "process_retinal_analysis" in src


def test_task_1_2_pdf_report_contains_images_and_table(tmp_path):
    """Task 1.2: pdf_generator.py renders PDF with original image, color overlay image, and lesion breakdown."""
    from app.services.pdf_generator import generate_medical_pdf_report

    orig_path = str(tmp_path / "orig.png")
    overlay_path = str(tmp_path / "overlay.png")
    pdf_path = str(tmp_path / "report.pdf")

    img_arr = np.full((128, 128, 3), 120, dtype=np.uint8)
    Image.fromarray(img_arr).save(orig_path)
    masks = np.zeros((5, 128, 128), dtype=np.float32)
    masks[0, 20:30, 20:30] = 1.0
    overlay_arr = render_color_mask_overlay(img_arr, masks, IDRID_CLASSES)
    Image.fromarray(overlay_arr).save(overlay_path)

    fake_patient = SimpleNamespace(
        full_name="Nguyễn Văn Kiểm Thử",
        patient_code="PAT-TEST-001",
        gender="Male",
        date_of_birth=date(1970, 5, 10),
    )
    fake_details = [
        SimpleNamespace(lesion_type="MA", count=3, area_pixels=100, area_percentage=0.12),
        SimpleNamespace(lesion_type="HE", count=1, area_pixels=80, area_percentage=0.09),
    ]
    fake_analysis = SimpleNamespace(
        id=999,
        task_type="IDRID_LESIONS",
        model_name="Attention_UNet",
        created_at=datetime(2026, 11, 26, 8, 0, 0),
        retinal_fov_pixels=80000,
        lesion_area_pixels=180,
        lesion_percentage=0.21,
        lesion_details=fake_details,
        dice_score=0.8652,
        iou_score=0.7624,
        precision=0.8775,
        recall=0.8532,
        specificity=0.9910,
        f1_score=0.8652,
    )

    out = generate_medical_pdf_report(
        analysis=fake_analysis,
        patient=fake_patient,
        doctor_name="Dr. Test",
        output_pdf_path=pdf_path,
        original_img_path=orig_path,
        predicted_mask_path=overlay_path,
        doctor_notes="Hẹn tái khám sau 3 tháng.",
    )
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5000


def test_task_1_3_dashboard_reads_training_history_csv():
    """Task 1.3: Dashboard model-performance API reads real metrics from *_training_history.csv."""
    from app.api.dashboard import get_model_performance

    perf = get_model_performance(db=None, current_user=None)
    assert len(perf) == 2
    by_name = {item["model_name"]: item for item in perf}

    for model_key, csv_name in [
        ("UNet_Baseline", "idrid_unet_training_history.csv"),
        ("Attention_UNet", "idrid_attention_unet_training_history.csv"),
    ]:
        csv_path = os.path.join(CHECKPOINT_DIR, csv_name)
        assert os.path.exists(csv_path), f"Missing {csv_name}"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        best = max(rows, key=lambda r: float(r["val_dice"]))
        stat = by_name[model_key]
        assert np.isclose(stat["mean_dice"], round(float(best["val_dice"]), 4), atol=1e-4)
        assert np.isclose(stat["mean_iou"], round(float(best["val_iou"]), 4), atol=1e-4)
        assert np.isclose(stat["mean_precision"], round(float(best["val_precision"]), 4), atol=1e-4)
        assert np.isclose(stat["mean_recall"], round(float(best["val_recall"]), 4), atol=1e-4)
        assert stat["sample_count"] == len(rows)


# ==============================================================================
# 2.3. MEMBER 2 — HANH (AI Core, Colab Experiments & Studio UI)
# ==============================================================================


def test_task_2_1_focal_and_combo_focal_dice_loss():
    """Task 2.1: FocalLoss & ComboFocalDiceLoss compute valid scalar loss and gradients."""
    logits = torch.randn(2, 5, 64, 64, requires_grad=True)
    targets = (torch.rand(2, 5, 64, 64) > 0.9).float()

    focal = FocalLoss(alpha=0.75, gamma=2.0)
    combo = ComboFocalDiceLoss(focal_weight=0.5, dice_weight=0.5, alpha=0.75, gamma=2.0)

    f_val = focal(logits, targets)
    c_val = combo(logits, targets)
    assert f_val.item() > 0
    assert c_val.item() > 0
    c_val.backward()
    assert logits.grad is not None


def test_task_2_1_clahe_and_512_resolution():
    """Task 2.1: 512x512 resolution and green-channel CLAHE enabled across datasets & inference."""
    assert IMAGE_SIZE == (512, 512), f"Expected IMAGE_SIZE=(512, 512), got {IMAGE_SIZE}"

    ds_idrid = IDRiDDataset(split="train")
    assert ds_idrid.apply_clahe is True
    assert ds_idrid.target_size == (512, 512)
    img_t, mask_t, _ = ds_idrid[0]
    assert img_t.shape == (3, 512, 512)
    assert mask_t.shape == (5, 512, 512)

    ds_drive = DRIVEDataset(split="train")
    assert ds_drive.apply_clahe is True
    assert ds_drive.target_size == (512, 512)

    predictor = RetinalPredictor(task_type="IDRID_LESIONS", model_name="Attention_UNet")
    raw = Image.new("RGB", (640, 480), color=(120, 80, 40))
    tensor_in, rgb_512 = predictor.preprocess_image(raw)
    assert tensor_in.shape == (1, 3, 512, 512)
    assert rgb_512.shape == (512, 512, 3)


def test_task_2_2_checkpoints_weights_and_csv_exist():
    """Task 2.2: Trained .pth weights and CSV logs exist for idrid_unet, idrid_attention_unet, and drive_unet."""
    required_files = [
        "idrid_unet_best_model.pth",
        "idrid_attention_unet_best.pth",
        "drive_unet_best.pth",
        "idrid_unet_training_history.csv",
        "idrid_attention_unet_training_history.csv",
        "drive_unet_training_history.csv",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(CHECKPOINT_DIR, f))]
    assert not missing, f"Missing checkpoint weights or CSV files in ai/checkpoints/: {missing}"


def test_task_2_2_training_curve_images_exist_for_all_3_models():
    """Task 2.2: Dice/IoU/Loss curve PNGs exist for all 3 models (idrid_unet, idrid_attention_unet, drive_unet)."""
    expected_curves = []
    for prefix in ["idrid_unet", "idrid_attention_unet", "drive_unet"]:
        for metric in ["dice_curve.png", "iou_curve.png", "loss_curve.png"]:
            expected_curves.append(f"{prefix}_{metric}")

    missing = [f for f in expected_curves if not os.path.exists(os.path.join(CHECKPOINT_DIR, f))]
    assert not missing, f"Missing training curve PNGs in ai/checkpoints/: {missing}"


def test_task_2_3_per_class_threshold_tuning_and_detection():
    """Task 2.3: Per-class threshold tuning and >0.5 detection across IDRiD lesions & DRIVE vessels."""
    predict_py = os.path.join(PROJECT_ROOT, "ai", "inference", "predict.py")
    with open(predict_py, "r", encoding="utf-8") as f:
        src = f.read()

    # Check whether per-class thresholds (or threshold tuning config) exist instead of only flat > 0.5
    has_per_class_threshold = bool(
        re.search(r"THRESHOLDS|class_thresholds|thresholds\s*[=:]", src, re.IGNORECASE)
    )

    pred_idrid = RetinalPredictor(task_type="IDRID_LESIONS", model_name="Attention_UNet")
    res_idrid = pred_idrid.predict(
        os.path.join(PROJECT_ROOT, "data", "IDRiD", "1. Original Images", "b. Testing Set", "IDRiD_55.jpg")
    )
    idrid_max_probs = {c: float(res_idrid["raw_probabilities"][i].max()) for i, c in enumerate(IDRID_CLASSES)}
    idrid_counts = {x["lesion_type"]: x["count"] for x in res_idrid["quantification"]["lesion_breakdown"]}

    pred_drive = RetinalPredictor(task_type="DRIVE_VESSELS", model_name="UNet_Baseline")
    res_drive = pred_drive.predict(
        os.path.join(PROJECT_ROOT, "data", "DRIVE", "test", "images", "01_test.tif"),
        fov_mask_path=os.path.join(PROJECT_ROOT, "data", "DRIVE", "test", "mask", "01_test_mask.gif"),
    )
    drive_max_prob = float(res_drive["raw_probabilities"][0].max())
    drive_vessel_count = res_drive["quantification"]["lesion_breakdown"][0]["count"]

    failures = []
    if not has_per_class_threshold:
        failures.append("ai/inference/predict.py still hardcodes single global threshold (probs > 0.5) without per-class threshold tuning.")
    low_idrid = {k: round(v, 4) for k, v in idrid_max_probs.items() if v <= 0.5}
    if low_idrid:
        failures.append(f"IDRiD max probabilities <= 0.5 for classes {low_idrid}; counts={idrid_counts}")
    if drive_max_prob <= 0.5 or drive_vessel_count == 0:
        failures.append(f"DRIVE VESSEL max_prob={drive_max_prob:.4f} (<= 0.5) -> detected count={drive_vessel_count}")

    assert not failures, " | ".join(failures)


def test_task_2_3_canvas_viewer_opacity_and_layer_filtering():
    """Task 2.3: CanvasViewer.tsx supports both Opacity control and actual per-class layer visibility filtering."""
    viewer_path = os.path.join(PROJECT_ROOT, "frontend", "src", "components", "CanvasViewer.tsx")
    with open(viewer_path, "r", encoding="utf-8") as f:
        src = f.read()

    # Opacity slider check
    assert "globalAlpha = opacity" in src, "CanvasViewer draw() must apply opacity via ctx.globalAlpha"

    # Extract draw() callback body to verify `classes` toggle state actually affects pixel/mask rendering
    draw_match = re.search(r"const draw = useCallback\(\(\) => \{(.*?)\}, \[[^\]]*\]\);", src, re.DOTALL)
    assert draw_match is not None, "Could not locate draw() callback in CanvasViewer.tsx"
    draw_body = draw_match.group(1)

    assert "classes" in draw_body, (
        "CanvasViewer.tsx defines `classes` toggle buttons, but `draw()` does not read `classes` "
        "to filter individual MA/HE/EX/SE/OD/VESSEL color layers on the rendered overlay."
    )
