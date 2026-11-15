from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel

class PatientBase(BaseModel):
    patient_code: str
    full_name: str
    gender: str
    date_of_birth: Optional[date] = None
    phone_number: Optional[str] = None
    medical_history: Optional[str] = None
    notes: Optional[str] = None

class PatientCreate(BaseModel):
    patient_code: Optional[str] = None # Auto-generated if omitted
    full_name: str
    gender: str
    date_of_birth: Optional[date] = None
    phone_number: Optional[str] = None
    medical_history: Optional[str] = None
    notes: Optional[str] = None

class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    phone_number: Optional[str] = None
    medical_history: Optional[str] = None
    notes: Optional[str] = None

class PatientOut(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime
    total_analyses: Optional[int] = 0

    class Config:
        from_attributes = True

class PatientListResponse(BaseModel):
    items: List[PatientOut]
    total: int
