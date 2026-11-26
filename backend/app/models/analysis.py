import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from app.database.session import Base

class AnalysisStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskType(str, enum.Enum):
    IDRID_LESIONS = "IDRID_LESIONS"
    DRIVE_VESSELS = "DRIVE_VESSELS"

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    task_type = Column(String(50), default=TaskType.IDRID_LESIONS.value, nullable=False)
    model_name = Column(String(100), default="UNet_Baseline", nullable=False)
    analysis_status = Column(String(20), default=AnalysisStatus.PENDING.value, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Image Paths & Storage Keys
    original_image_url = Column(String(500), nullable=False)
    predicted_mask_url = Column(String(500), nullable=True)
    ground_truth_mask_url = Column(String(500), nullable=True)
    fov_mask_url = Column(String(500), nullable=True)

    # Medical Quantification Results
    retinal_fov_pixels = Column(Integer, default=0, nullable=True)
    lesion_area_pixels = Column(Integer, default=0, nullable=True)
    lesion_percentage = Column(Float, default=0.0, nullable=True)
    processing_time_ms = Column(Integer, default=0, nullable=True)

    # Model Evaluation Metrics (Evaluated if Ground Truth is provided)
    dice_score = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="analyses")
    doctor = relationship("User", back_populates="analyses")
    lesion_details = relationship("LesionDetail", back_populates="analysis", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="analysis", cascade="all, delete-orphan")

    @property
    def patient_code(self) -> str | None:
        return self.patient.patient_code if self.patient else None

    @property
    def patient_name(self) -> str | None:
        return self.patient.full_name if self.patient else None


class LesionDetail(Base):
    __tablename__ = "lesion_details"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    
    lesion_type = Column(String(30), nullable=False) # 'MA', 'HE', 'EX', 'SE', 'OD', 'VESSEL'
    count = Column(Integer, default=0, nullable=False)
    area_pixels = Column(Integer, default=0, nullable=False)
    area_percentage = Column(Float, default=0.0, nullable=False)

    analysis = relationship("Analysis", back_populates="lesion_details")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    report_code = Column(String(100), unique=True, index=True, nullable=False)
    pdf_path = Column(String(500), nullable=False)
    doctor_notes = Column(Text, nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="reports")

