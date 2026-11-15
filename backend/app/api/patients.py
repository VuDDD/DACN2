from datetime import datetime
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc

from app.database.session import get_db
from app.models.patient import Patient
from app.models.user import User
from app.schemas.patient import PatientCreate, PatientUpdate, PatientOut, PatientListResponse
from app.api.deps import get_current_user

router = APIRouter(prefix="/patients", tags=["Patient Management"])

def generate_patient_code(db: Session) -> str:
    year = datetime.utcnow().year
    count = db.query(Patient).count() + 1
    return f"PAT-{year}-{count:04d}"

@router.get("", response_model=PatientListResponse)
def list_patients(
    search: Optional[str] = Query(None, description="Search by name, code or phone"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Retrieve list of patients with search and pagination."""
    query = db.query(Patient)
    if search:
        s = f"%{search}%"
        query = query.filter(
            or_(
                Patient.full_name.ilike(s),
                Patient.patient_code.ilike(s),
                Patient.phone_number.ilike(s)
            )
        )
    total = query.count()
    patients = query.order_by(desc(Patient.created_at)).offset(skip).limit(limit).all()
    
    # Attach total_analyses count
    result_items = []
    for p in patients:
        out = PatientOut.from_orm(p)
        out.total_analyses = len(p.analyses)
        result_items.append(out)

    return {"items": result_items, "total": total}

@router.post("", response_model=PatientOut, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Create a new patient record."""
    code = patient_in.patient_code or generate_patient_code(db)
    
    # Check duplicate code
    existing = db.query(Patient).filter(Patient.patient_code == code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient with code '{code}' already exists."
        )

    patient = Patient(
        patient_code=code,
        full_name=patient_in.full_name,
        gender=patient_in.gender,
        date_of_birth=patient_in.date_of_birth,
        phone_number=patient_in.phone_number,
        medical_history=patient_in.medical_history,
        notes=patient_in.notes
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    
    out = PatientOut.from_orm(patient)
    out.total_analyses = 0
    return out

@router.get("/{id}", response_model=PatientOut)
def get_patient(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Get single patient details by ID."""
    patient = db.query(Patient).filter(Patient.id == id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    out = PatientOut.from_orm(patient)
    out.total_analyses = len(patient.analyses)
    return out

@router.put("/{id}", response_model=PatientOut)
def update_patient(
    id: int,
    patient_in: PatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Update patient information."""
    patient = db.query(Patient).filter(Patient.id == id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    update_data = patient_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)

    db.add(patient)
    db.commit()
    db.refresh(patient)
    out = PatientOut.from_orm(patient)
    out.total_analyses = len(patient.analyses)
    return out

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> None:
    """Delete a patient record and their associated analyses."""
    patient = db.query(Patient).filter(Patient.id == id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    db.delete(patient)
    db.commit()
