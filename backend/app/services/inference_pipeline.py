import sys
import os
import io
import time
import logging
from PIL import Image
from sqlalchemy.orm import Session

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models.analysis import Analysis, LesionDetail, AnalysisStatus, TaskType
from app.storage.service import storage_service
from ai.inference.predict import RetinalPredictor

logger = logging.getLogger("inference_pipeline")

def process_retinal_analysis(analysis_id: int, db: Session) -> Analysis:
    """
    Executes end-to-end retinal AI analysis:
    1. Loads patient retinal photo
    2. Runs Deep Learning model (U-Net or Attention U-Net)
    3. Analyzes Connected Components & FOV Lesion Density
    4. Persists visual mask overlay into storage
    5. Updates analysis status and metric findings in DB
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise ValueError(f"Analysis with ID {analysis_id} not found.")

    analysis.analysis_status = AnalysisStatus.PROCESSING.value
    db.commit()

    try:
        abs_img = storage_service.get_absolute_path(analysis.original_image_url)
        if not abs_img or not os.path.exists(abs_img):
            raise FileNotFoundError(f"Input image not found: {analysis.original_image_url}")

        abs_gt = storage_service.get_absolute_path(analysis.ground_truth_mask_url) if analysis.ground_truth_mask_url else None
        abs_fov = storage_service.get_absolute_path(analysis.fov_mask_url) if analysis.fov_mask_url else None

        # Instantiate Predictor
        predictor = RetinalPredictor(
            task_type=analysis.task_type,
            model_name=analysis.model_name
        )

        # Run inference
        res = predictor.predict(
            image_path=abs_img,
            ground_truth_path=abs_gt,
            fov_mask_path=abs_fov
        )

        # Save generated overlay image to storage
        overlay_rgb = res["overlay_image"]
        overlay_pil = Image.fromarray(overlay_rgb)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        overlay_bytes = buf.getvalue()

        overlay_fn = f"overlay_{analysis.id}_{int(time.time())}.png"
        rel_overlay_url, _ = storage_service.save_bytes(overlay_bytes, filename=overlay_fn, subfolder="overlays")

        # Update quantification findings
        quant = res["quantification"]
        analysis.retinal_fov_pixels = quant["retinal_fov_pixels"]
        analysis.lesion_area_pixels = quant["total_lesion_pixels"]
        analysis.lesion_percentage = quant["total_lesion_percentage"]
        analysis.processing_time_ms = res["processing_time_ms"]
        analysis.predicted_mask_url = rel_overlay_url

        # Record validation metrics if ground truth was present
        if res["metrics"]:
            m = res["metrics"]
            analysis.dice_score = m.get("dice")
            analysis.iou_score = m.get("iou")
            analysis.precision = m.get("precision")
            analysis.recall = m.get("recall")
            analysis.specificity = m.get("specificity")
            analysis.f1_score = m.get("f1")

        # Clear existing lesion details if re-analyzing
        db.query(LesionDetail).filter(LesionDetail.analysis_id == analysis.id).delete()

        # Save individual lesion breakdown
        for item in quant["lesion_breakdown"]:
            detail = LesionDetail(
                analysis_id=analysis.id,
                lesion_type=item["lesion_type"],
                count=item["count"],
                area_pixels=item["area_pixels"],
                area_percentage=item["area_percentage"]
            )
            db.add(detail)

        analysis.analysis_status = AnalysisStatus.COMPLETED.value
        analysis.error_message = None
        db.commit()
        db.refresh(analysis)
        logger.info(f"Analysis {analysis.id} completed successfully in {res['processing_time_ms']}ms.")
        return analysis

    except Exception as e:
        logger.error(f"Error processing analysis {analysis.id}: {e}", exc_info=True)
        analysis.analysis_status = AnalysisStatus.FAILED.value
        analysis.error_message = str(e)
        db.commit()
        raise

