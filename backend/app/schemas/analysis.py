from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, field_validator


def _coerce_sqlite_int(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return int.from_bytes(v, byteorder="little", signed=True)
    return v


class LesionDetailOut(BaseModel):
    id: int
    lesion_type: str
    count: int
    area_pixels: int
    area_percentage: float

    @field_validator("count", "area_pixels", mode="before")
    @classmethod
    def decode_legacy_bytes(cls, v: Any) -> Any:
        return _coerce_sqlite_int(v)

    class Config:
        from_attributes = True

class AnalysisBase(BaseModel):
    patient_id: int
    task_type: str = "IDRID_LESIONS"
    model_name: str = "UNet_Baseline"

class AnalysisCreate(AnalysisBase):
    pass

class AnalysisOut(BaseModel):
    id: int
    patient_id: int
    patient_code: Optional[str] = None
    patient_name: Optional[str] = None
    user_id: Optional[int] = None
    task_type: str
    model_name: str
    analysis_status: str
    error_message: Optional[str] = None
    
    original_image_url: str
    predicted_mask_url: Optional[str] = None
    ground_truth_mask_url: Optional[str] = None
    fov_mask_url: Optional[str] = None

    retinal_fov_pixels: Optional[int] = 0
    lesion_area_pixels: Optional[int] = 0
    lesion_percentage: Optional[float] = 0.0
    processing_time_ms: Optional[int] = 0

    @field_validator("retinal_fov_pixels", "lesion_area_pixels", "processing_time_ms", mode="before")
    @classmethod
    def decode_legacy_int_bytes(cls, v: Any) -> Any:
        return _coerce_sqlite_int(v)

    dice_score: Optional[float] = None
    iou_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    f1_score: Optional[float] = None

    created_at: datetime
    updated_at: datetime
    lesion_details: List[LesionDetailOut] = []

    class Config:
        from_attributes = True

class AnalysisStatusResponse(BaseModel):
    id: int
    status: str
    progress: Optional[int] = 0
    message: Optional[str] = None
    completed: bool = False

class AnalysisListResponse(BaseModel):
    items: List[AnalysisOut]
    total: int

