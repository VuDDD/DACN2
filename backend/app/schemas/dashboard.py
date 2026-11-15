from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.schemas.analysis import AnalysisOut

class DashboardStatistics(BaseModel):
    total_patients: int
    total_analyses: int
    completed_analyses: int
    processing_analyses: int
    failed_analyses: int
    average_processing_time_ms: float
    total_lesions_detected: int

class LesionDistributionStat(BaseModel):
    lesion_type: str
    label: str
    total_count: int
    total_area_pixels: int
    color: str

class ModelPerformanceStat(BaseModel):
    model_name: str
    task_type: str
    mean_dice: float
    mean_iou: float
    mean_precision: float
    mean_recall: float
    sample_count: int

