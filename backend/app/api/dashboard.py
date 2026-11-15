from typing import Any, List, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.database.session import get_db
from app.models.patient import Patient
from app.models.analysis import Analysis, LesionDetail, AnalysisStatus
from app.models.user import User
from app.schemas.dashboard import DashboardStatistics, LesionDistributionStat, ModelPerformanceStat
from app.schemas.analysis import AnalysisOut
from app.api.deps import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard & Statistics"])

@router.get("/statistics", response_model=DashboardStatistics)
def get_dashboard_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Retrieve high-level operational statistics for the dashboard."""
    total_patients = db.query(Patient).count()
    total_analyses = db.query(Analysis).count()
    completed_analyses = db.query(Analysis).filter(Analysis.analysis_status == AnalysisStatus.COMPLETED.value).count()
    processing_analyses = db.query(Analysis).filter(
        Analysis.analysis_status.in_([AnalysisStatus.PENDING.value, AnalysisStatus.PROCESSING.value])
    ).count()
    failed_analyses = db.query(Analysis).filter(Analysis.analysis_status == AnalysisStatus.FAILED.value).count()

    # Average processing time for completed
    avg_time = db.query(func.avg(Analysis.processing_time_ms)).filter(
        Analysis.analysis_status == AnalysisStatus.COMPLETED.value
    ).scalar() or 0.0

    # Total abnormal lesions detected count
    total_lesions = db.query(func.sum(LesionDetail.count)).filter(
        LesionDetail.lesion_type.in_(["MA", "HE", "EX", "SE"])
    ).scalar() or 0

    return {
        "total_patients": total_patients,
        "total_analyses": total_analyses,
        "completed_analyses": completed_analyses,
        "processing_analyses": processing_analyses,
        "failed_analyses": failed_analyses,
        "average_processing_time_ms": float(avg_time),
        "total_lesions_detected": int(total_lesions)
    }

@router.get("/recent-analyses", response_model=List[AnalysisOut])
def get_recent_analyses(
    limit: int = 8,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Retrieve most recent analyses for the activity feed."""
    analyses = db.query(Analysis).order_by(desc(Analysis.created_at)).limit(limit).all()
    return analyses

@router.get("/lesion-distribution", response_model=List[LesionDistributionStat])
def get_lesion_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Aggregates lesion counts and area distribution across all completed diagnoses."""
    meta = {
        "MA": ("Microaneurysms", "#ef4444"), # Red
        "HE": ("Haemorrhages", "#f97316"),   # Orange
        "EX": ("Hard Exudates", "#eab308"),  # Yellow
        "SE": ("Soft Exudates", "#22c55e"),  # Green
        "OD": ("Optic Disc", "#3b82f6"),     # Blue
        "VESSEL": ("Blood Vessels", "#a855f7") # Purple
    }

    results = []
    for l_type, (label, color) in meta.items():
        q = db.query(
            func.sum(LesionDetail.count).label("cnt"),
            func.sum(LesionDetail.area_pixels).label("area")
        ).filter(LesionDetail.lesion_type == l_type).first()
        
        cnt = q.cnt or 0
        area = q.area or 0
        results.append({
            "lesion_type": l_type,
            "label": label,
            "total_count": int(cnt),
            "total_area_pixels": int(area),
            "color": color
        })

    return results

@router.get("/model-performance", response_model=List[ModelPerformanceStat])
def get_model_performance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Returns experimental validation metrics across AI models."""
    models = ["UNet_Baseline", "Attention_UNet"]
    stats = []
    
    for m in models:
        q = db.query(
            func.avg(Analysis.dice_score).label("avg_dice"),
            func.avg(Analysis.iou_score).label("avg_iou"),
            func.avg(Analysis.precision).label("avg_prec"),
            func.avg(Analysis.recall).label("avg_rec"),
            func.count(Analysis.id).label("cnt")
        ).filter(Analysis.model_name == m, Analysis.dice_score.isnot(None)).first()

        cnt = q.cnt or 0
        # If no verified benchmark run yet in DB, report 0.0 with sample_count 0 (strict no-fake rule)
        stats.append({
            "model_name": m,
            "task_type": "Multi-Lesion & Vessel",
            "mean_dice": float(q.avg_dice) if q.avg_dice is not None else 0.0,
            "mean_iou": float(q.avg_iou) if q.avg_iou is not None else 0.0,
            "mean_precision": float(q.avg_prec) if q.avg_prec is not None else 0.0,
            "mean_recall": float(q.avg_rec) if q.avg_rec is not None else 0.0,
            "sample_count": cnt
        })

    return stats
