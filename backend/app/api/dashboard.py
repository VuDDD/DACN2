import os
import csv
from typing import Any, List, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc

from app.database.session import get_db
from app.models.patient import Patient
from app.models.analysis import Analysis, LesionDetail, AnalysisStatus
from app.models.user import User
from app.schemas.dashboard import DashboardStatistics, LesionDistributionStat, ModelPerformanceStat
from app.schemas.analysis import AnalysisOut
from app.api.deps import get_current_user

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "ai", "checkpoints")

CSV_MAPPING = {
    "UNet_Baseline": ["idrid_unet_training_history.csv", "unet_training_history.csv"],
    "Attention_UNet": ["idrid_attention_unet_training_history.csv", "attention_unet_training_history.csv"],
}

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
    analyses = db.query(Analysis).options(joinedload(Analysis.patient)).order_by(desc(Analysis.created_at)).limit(limit).all()
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
    """Returns experimental validation metrics across AI models by parsing training history logs."""
    models = ["UNet_Baseline", "Attention_UNet"]
    stats = []
    
    for m in models:
        # Check training history CSV files first
        csv_candidates = CSV_MAPPING.get(m, [])
        csv_metrics = None
        sample_count = 0

        for fn in csv_candidates:
            p = os.path.join(CHECKPOINT_DIR, fn)
            if os.path.exists(p):
                try:
                    with open(p, mode="r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            sample_count = len(rows)
                            # Select highest validation performance epoch by val_dice
                            best_row = max(rows, key=lambda r: float(r.get("val_dice", 0.0) or 0.0))
                            d = float(best_row.get("val_dice", 0.0) or 0.0)
                            iou = float(best_row.get("val_iou", 0.0) or 0.0)
                            prec = float(best_row.get("val_precision", 0.0) or ((2 * iou) / (1 + iou) if (1 + iou) > 0 else 0.0))
                            rec = float(best_row.get("val_recall", 0.0) or d)
                            csv_metrics = {
                                "mean_dice": round(d, 4),
                                "mean_iou": round(iou, 4),
                                "mean_precision": round(prec, 4),
                                "mean_recall": round(rec, 4),
                                "sample_count": sample_count
                            }
                            break
                except Exception:
                    pass

        if csv_metrics:
            stats.append({
                "model_name": m,
                "task_type": "Multi-Lesion & Vessel",
                "mean_dice": csv_metrics["mean_dice"],
                "mean_iou": csv_metrics["mean_iou"],
                "mean_precision": csv_metrics["mean_precision"],
                "mean_recall": csv_metrics["mean_recall"],
                "sample_count": csv_metrics["sample_count"]
            })
        else:
            q = db.query(
                func.avg(Analysis.dice_score).label("avg_dice"),
                func.avg(Analysis.iou_score).label("avg_iou"),
                func.avg(Analysis.precision).label("avg_prec"),
                func.avg(Analysis.recall).label("avg_rec"),
                func.count(Analysis.id).label("cnt")
            ).filter(Analysis.model_name == m, Analysis.dice_score.isnot(None)).first()

            cnt = q.cnt or 0
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

