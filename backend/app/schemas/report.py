from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ReportBase(BaseModel):
    analysis_id: int
    doctor_notes: Optional[str] = None

class ReportCreate(ReportBase):
    pass

class ReportOut(BaseModel):
    id: int
    analysis_id: int
    report_code: str
    pdf_path: str
    doctor_notes: Optional[str] = None
    generated_at: datetime

    class Config:
        from_attributes = True
