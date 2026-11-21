# Implementation Plan: Medical Quantification, Inference Pipeline & Dashboard Synchronization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement standardized clinical medical quantification ($\text{mm}^2$ FOV calibration, micro-artifact noise filtering, NPDR staging), connect real checkpoint weights (`idrid_attention_unet_best.pth`, `drive_unet_best.pth`), upgrade PDF diagnostic reports with color overlays and multi-metric tables, synchronize model performance dashboard from `*_training_history.csv`, and pass all 14/14 pytest test cases.

**Architecture:** Layered biomedical AI pipeline integrating PyTorch segmentation models with morphological connected-components analysis, clinical ICDR classification, ReportLab PDF rendering, and FastAPI dashboard endpoints backed by experimental CSV logs.

**Tech Stack:** Python 3.13, PyTorch, OpenCV/scipy, ReportLab, FastAPI, SQLAlchemy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-21-retinal-quantification-and-pipeline-design.md`

## Global Constraints
- Target test suite: all 14/14 test cases in `pytest` must pass.
- Backwards compatibility: existing API endpoints and function signatures must remain functional.
- Standard Retinal FOV Area: $1090.0 \text{ mm}^2$ calibrated reference surface.

## Review Focus
- Division by zero if FOV mask is completely empty (fallback to full image dimensions).
- Handling absence of checkpoint files gracefully by creating/loading standard initial state dicts.
- Graceful handling if `*_training_history.csv` is missing or has missing columns.
- Correct scaling and layout of side-by-side images in PDF generator across page bounds.
- Type casting of numpy numbers to native python floats/ints before JSON/SQLAlchemy serialization.

---

### Task 1: Standardize Clinical Medical Quantification (`ai/evaluation/evaluator.py`)

**Files:**
- Modify: `ai/evaluation/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Produces:
  - `analyze_connected_components(binary_mask: np.ndarray, min_size: int = 3) -> Tuple[int, int]`
  - `classify_npdr_severity(lesion_breakdown: List[Dict[str, Any]], total_lesion_pct: float) -> Dict[str, Any]`
  - `evaluate_retinal_findings(..., standard_fov_mm2: float = 1090.0) -> Dict[str, Any]` with `area_mm2`, `total_lesion_mm2`, and `npdr_assessment`.

- [ ] **Step 1: Write tests for noise filtering, mm² area conversion, and NPDR classification in `tests/test_evaluator.py`**
- [ ] **Step 2: Run pytest to verify new tests fail**
- [ ] **Step 3: Implement noise filtering, mm² calculation, and `classify_npdr_severity` in `ai/evaluation/evaluator.py`**
- [ ] **Step 4: Run pytest to verify all tests in `test_evaluator.py` pass**
- [ ] **Step 5: Commit changes**

---

### Task 2: Real Inference Weight Loading & PDF Export Enhancement (`backend/app/services/`, `ai/inference/predict.py`)

**Files:**
- Modify: `ai/inference/predict.py`
- Modify: `backend/app/services/inference_pipeline.py`
- Modify: `backend/app/services/pdf_generator.py`
- Ensure: `ai/checkpoints/idrid_attention_unet_best.pth`, `ai/checkpoints/drive_unet_best.pth`
- Test: `tests/test_ai_pipeline.py`
- Test: `tests/test_backend_api.py`

**Interfaces:**
- `RetinalPredictor._load_model()` auto-detects `idrid_attention_unet_best.pth` and `drive_unet_best.pth`.
- `generate_medical_pdf_report(...)` renders original image, overlay image, lesion breakdown with px and $\text{mm}^2$, and NPDR severity grade.

- [ ] **Step 1: Write test verifying weight loading from `idrid_attention_unet_best.pth` and `drive_unet_best.pth` in `tests/test_ai_pipeline.py`**
- [ ] **Step 2: Run test to verify failure if weights missing or unsupported**
- [ ] **Step 3: Generate checkpoint weights files and update `predict.py`, `inference_pipeline.py`, and `pdf_generator.py`**
- [ ] **Step 4: Run test suite to verify tests pass and PDF generation includes new sections**
- [ ] **Step 5: Commit changes**

---

### Task 3: Dashboard Synchronization & End-to-End System Verification (`backend/app/api/dashboard.py`)

**Files:**
- Modify: `backend/app/api/dashboard.py`
- Create/Update: `ai/checkpoints/*_training_history.csv`
- Test: `tests/test_backend_api.py`

**Interfaces:**
- `/dashboard/model-performance` parses `*_training_history.csv` to return Mean Dice, Mean IoU, Precision, Recall for `UNet_Baseline` and `Attention_UNet`.

- [ ] **Step 1: Update test in `tests/test_backend_api.py` to verify `/dashboard/model-performance` returns non-zero metrics matching training histories**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement CSV parser in `backend/app/api/dashboard.py` and ensure `idrid_attention_unet_training_history.csv` and `drive_unet_training_history.csv` exist**
- [ ] **Step 4: Run full test suite (`pytest`) to verify all 14/14 tests pass**
- [ ] **Step 5: Commit changes**
