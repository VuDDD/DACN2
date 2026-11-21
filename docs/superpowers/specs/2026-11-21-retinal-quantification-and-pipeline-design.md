# Medical Quantification, Real Inference Pipeline & Dashboard Synchronization Design

## 1. Context and Goals
The objective is to implement:
- **Task 1.1**: Standardize clinical medical quantification in `ai/evaluation/evaluator.py`: connected components analysis with micro-artifact noise filtering (< 3-5 px), lesion surface conversion from pixel counts to calibrated physical metric $\text{mm}^2$ based on retinal Field of View (FOV mask), and clinical disease severity grading for Diabetic Retinopathy (No DR, Mild, Moderate, Severe NPDR) conforming to the International Clinical Diabetic Retinopathy (ICDR) scale.
- **Task 1.2**: Connect real inference pipeline in `backend/app/services/inference_pipeline.py` and `ai/inference/predict.py`: auto-load model weights `idrid_attention_unet_best.pth` and `drive_unet_best.pth` from `ai/checkpoints/` with clean architecture handling and fallback. Upgrade `backend/app/services/pdf_generator.py` to produce a high-fidelity diagnostic report containing the original fundus photograph, translucent multi-color segmentation overlay, detailed lesion metric breakdown table (foci count, px, $\text{mm}^2$, $\%$ FOV), and clinical severity assessment.
- **Task 1.3**: Synchronize Dashboard in `backend/app/api/dashboard.py` by reading model benchmark metrics (Mean Dice, Mean IoU, Precision, Recall) directly from `*_training_history.csv` in `ai/checkpoints/` comparing U-Net Baseline vs Attention U-Net. Verify 100% acceptance across all pytest test suites.

## 2. Technical Architecture & Calculations
### 2.1 Optical & Spatial Retinal FOV Area Calibration
- For standard clinical 45° to 50° fundus cameras (such as those used in IDRiD and DRIVE datasets), the human retina has an average visible surface area of $A_{\text{standard\_fov}} \approx 1090.0 \text{ mm}^2$.
- Given the detected retinal mask with $N_{\text{fov}}$ valid pixels and a lesion mask of $N_{\text{lesion}}$ pixels:
  $$\text{Area}_{\text{mm}^2} = \left(\frac{N_{\text{lesion}}}{\max(1, N_{\text{fov}})}\right) \times 1090.0$$

### 2.2 Clinical NPDR Staging (ICDR Scale)
- **No DR**: No microaneurysms ($N_{\text{MA}} = 0$) and no hemorrhages/exudates.
- **Mild NPDR**: Microaneurysms only ($N_{\text{MA}} > 0$), $N_{\text{HE}} \le 2$ or lesion ratio $< 0.05\%$.
- **Moderate NPDR**: Presence of MA and moderate HE, Hard Exudates (EX), or Soft Exudates (SE), but below Severe criteria.
- **Severe NPDR**: Extensive hemorrhages ($N_{\text{HE}} \ge 15$ or HE area $> 0.5\%$), extensive cotton-wool patches ($N_{\text{SE}} \ge 5$), or total pathological lesion area $> 1.5\%$ of FOV.

### 2.3 Weight Checkpoint Specification
- Paths:
  - `ai/checkpoints/idrid_attention_unet_best.pth` (Attention U-Net, 5 classes: MA, HE, EX, SE, OD)
  - `ai/checkpoints/drive_unet_best.pth` (U-Net, 1 class: Vessel)
- Auto-loader detects either `*_best.pth` or `*_best_model.pth`.

### 2.4 Training History CSV Synchronization
- Files:
  - `ai/checkpoints/idrid_unet_training_history.csv`
  - `ai/checkpoints/idrid_attention_unet_training_history.csv`
  - `ai/checkpoints/drive_unet_training_history.csv`
- Dashboard parses the highest validation metric epoch to compute Mean Dice, Mean IoU, Mean Precision, Mean Recall for model performance comparison.
