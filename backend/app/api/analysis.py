import os
import time
import logging
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from pydantic import BaseModel

from app.database.session import get_db
from app.models.analysis import Analysis, LesionDetail, AnalysisStatus, TaskType
from app.models.patient import Patient
from app.models.user import User
from app.schemas.analysis import AnalysisCreate, AnalysisOut, AnalysisStatusResponse, AnalysisListResponse
from app.api.deps import get_current_user
from app.storage.service import storage_service

logger = logging.getLogger("analysis_api")
router = APIRouter(prefix="/analysis", tags=["AI Analysis & Inference"])

class PredictRequest(BaseModel):
    patient_id: int
    image_url: str
    task_type: Optional[str] = TaskType.IDRID_LESIONS.value
    model_name: Optional[str] = "UNet_Baseline"
    ground_truth_url: Optional[str] = None
    fov_mask_url: Optional[str] = None

def run_in_process_analysis(analysis_id: int):
    """
    In-process async analysis fallback when Celery/Redis is offline.
    Uses AI Inference Engine & Medical Evaluator.
    """
    from app.database.session import SessionLocal
    db = SessionLocal()
    try:
        from app.services.inference_pipeline import process_retinal_analysis
        process_retinal_analysis(analysis_id, db)
    except Exception as e:
        logger.error(f"In-process analysis failed for analysis {analysis_id}: {e}", exc_info=True)
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.analysis_status = AnalysisStatus.FAILED.value
            analysis.error_message = str(e)
            db.commit()
    finally:
        db.close()

@router.post("/predict", response_model=AnalysisOut, status_code=status.HTTP_202_ACCEPTED)
def trigger_prediction(
    req: PredictRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Dispatches a retinal segmentation and quantification task."""
    patient = db.query(Patient).filter(Patient.id == req.patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Validate image accessibility
    abs_img = storage_service.get_absolute_path(req.image_url)
    if not abs_img or not os.path.exists(abs_img):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provided image_url cannot be resolved.")

    analysis = Analysis(
        patient_id=req.patient_id,
        user_id=current_user.id,
        task_type=req.task_type or TaskType.IDRID_LESIONS.value,
        model_name=req.model_name or "UNet_Baseline",
        analysis_status=AnalysisStatus.PENDING.value,
        original_image_url=req.image_url,
        ground_truth_mask_url=req.ground_truth_url,
        fov_mask_url=req.fov_mask_url
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Dispatch to Celery if available, otherwise FastAPI background task
    celery_dispatched = False
    try:
        from app.workers.celery_app import celery_app
        # Ping broker check
        conn = celery_app.connection()
        conn.connect()
        conn.release()
        from app.workers.tasks import celery_run_analysis
        celery_run_analysis.delay(analysis.id)
        celery_dispatched = True
        logger.info(f"Dispatched analysis {analysis.id} to Celery Worker.")
    except Exception as e:
        logger.info(f"Celery queue not connected ({e}). Running via FastAPI BackgroundTasks.")
        background_tasks.add_task(run_in_process_analysis, analysis.id)

    return analysis

@router.get("", response_model=AnalysisListResponse)
def list_analyses(
    patient_id: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """List historical analyses with optional patient filter."""
    query = db.query(Analysis).options(joinedload(Analysis.patient))
    if patient_id is not None:
        query = query.filter(Analysis.patient_id == patient_id)
    
    total = query.count()
    items = query.order_by(desc(Analysis.created_at)).offset(skip).limit(limit).all()
    return {"items": items, "total": total}

@router.get("/{id}", response_model=AnalysisOut)
def get_analysis_details(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Retrieve full analysis report including lesion breakdowns."""
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis

@router.get("/{id}/status", response_model=AnalysisStatusResponse)
def get_analysis_status(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Lightweight polling endpoint to track task lifecycle."""
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    progress = 0
    if analysis.analysis_status == AnalysisStatus.PENDING.value:
        progress = 10
    elif analysis.analysis_status == AnalysisStatus.PROCESSING.value:
        progress = 55
    elif analysis.analysis_status == AnalysisStatus.COMPLETED.value:
        progress = 100
    elif analysis.analysis_status == AnalysisStatus.FAILED.value:
        progress = 100

    return {
        "id": analysis.id,
        "status": analysis.analysis_status,
        "progress": progress,
        "message": analysis.error_message or f"Analysis is {analysis.analysis_status.lower()}",
        "completed": analysis.analysis_status in (AnalysisStatus.COMPLETED.value, AnalysisStatus.FAILED.value)
    }

@router.get("/{id}/result", response_model=AnalysisOut)
def get_analysis_result(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Retrieves final segmentation and quantification result."""
    analysis = db.query(Analysis).filter(Analysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis

