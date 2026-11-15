import os
import sys
import pytest
from datetime import date

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import app.core.cython_hook
from app.database.session import SessionLocal, Base, engine
from app.main import init_db
from app.models.user import User, UserRole
from app.models.patient import Patient
from app.models.analysis import Analysis, AnalysisStatus
from app.schemas.user import UserCreate
from app.schemas.patient import PatientCreate, PatientUpdate
from app.api.auth import register, login_json, LoginRequest
from app.api.patients import create_patient, list_patients, get_patient, update_patient, delete_patient
from app.api.images import list_sample_images, load_sample_into_storage
from app.api.dashboard import get_dashboard_statistics, get_lesion_distribution, get_model_performance
from app.api.analysis import trigger_prediction, get_analysis_details, get_analysis_status, PredictRequest
from app.api.reports import create_report, GenerateReportRequest

@pytest.fixture(scope="module")
def db_session():
    """Initializes tables and seeds initial administrator."""
    init_db()
    db = SessionLocal()
    yield db
    db.close()

@pytest.fixture(scope="module")
def admin_user(db_session):
    return db_session.query(User).filter(User.username == "admin").first()

def test_auth_registration_and_login(db_session):
    """Verify user registration and JWT login."""
    unique_user = "dr_ophthalmologist_vku"
    existing = db_session.query(User).filter(User.username == unique_user).first()
    if not existing:
        u_in = UserCreate(
            username=unique_user,
            email=f"{unique_user}@vku.vn",
            password="SecurePassword123!",
            full_name="Dr. Test Specialist",
            role="DOCTOR"
        )
        new_u = register(u_in, db=db_session)
        assert new_u.username == unique_user
        assert new_u.role == "DOCTOR"

    # Test login
    req = LoginRequest(username=unique_user, password="SecurePassword123!")
    login_out = login_json(req, db=db_session)
    assert "access_token" in login_out
    assert login_out["token_type"] == "bearer"
    assert login_out["user"].username == unique_user

def test_patient_management_lifecycle(db_session, admin_user):
    """Verify full CRUD lifecycle for patient records."""
    # Create Patient
    p_in = PatientCreate(
        full_name="Trần Thị Kiểm Thử",
        gender="Female",
        date_of_birth=date(1978, 6, 20),
        phone_number="0987654321",
        medical_history="Tiểu đường tuýp 2, thị lực mắt phải giảm sút",
        notes="Hẹn tái khám đáy mắt định kỳ"
    )
    p_out = create_patient(p_in, db=db_session, current_user=admin_user)
    assert p_out.id is not None
    assert p_out.patient_code.startswith("PAT-")
    assert p_out.full_name == "Trần Thị Kiểm Thử"

    # List Patients
    p_list = list_patients(search="Trần Thị", skip=0, limit=10, db=db_session, current_user=admin_user)
    assert p_list["total"] >= 1
    assert any(p.id == p_out.id for p in p_list["items"])

    # Get by ID
    p_detail = get_patient(p_out.id, db=db_session, current_user=admin_user)
    assert p_detail.id == p_out.id

    # Update Patient
    p_up = PatientUpdate(medical_history="Cập nhật: Đã điều trị laser quang đông võng mạc")
    p_updated = update_patient(p_out.id, p_up, db=db_session, current_user=admin_user)
    assert "laser quang đông" in p_updated.medical_history

def test_image_samples_and_load(db_session, admin_user):
    """Verify sample retrieval and loading into local storage."""
    samples = list_sample_images()
    assert len(samples) > 0, "Benchmark samples from IDRiD and DRIVE should be detected"
    
    # Load first sample into storage
    first = samples[0]
    loaded = load_sample_into_storage(dataset=first["dataset"], filename=first["filename"], current_user=admin_user)
    assert "url" in loaded
    assert loaded["width"] > 0
    assert loaded["height"] > 0

def test_dashboard_statistics_and_distributions(db_session, admin_user):
    """Verify dashboard metrics, lesion distribution, and model performance."""
    stats = get_dashboard_statistics(db=db_session, current_user=admin_user)
    assert stats["total_patients"] >= 1
    assert "completed_analyses" in stats

    dist = get_lesion_distribution(db=db_session, current_user=admin_user)
    assert len(dist) == 6 # MA, HE, EX, SE, OD, VESSEL
    
    perf = get_model_performance(db=db_session, current_user=admin_user)
    assert len(perf) == 2 # UNet_Baseline, Attention_UNet

def test_report_generation(db_session, admin_user):
    """Verify PDF report generation for an analysis record."""
    # Create mock completed analysis
    patient = db_session.query(Patient).first()
    assert patient is not None

    samples = list_sample_images()
    first = samples[0]
    loaded = load_sample_into_storage(dataset=first["dataset"], filename=first["filename"], current_user=admin_user)

    analysis = Analysis(
        patient_id=patient.id,
        user_id=admin_user.id,
        task_type="IDRID_LESIONS",
        model_name="Attention_UNet",
        analysis_status=AnalysisStatus.COMPLETED.value,
        original_image_url=loaded["url"],
        retinal_fov_pixels=100000,
        lesion_area_pixels=1500,
        lesion_percentage=1.5,
        processing_time_ms=250,
        dice_score=0.8523,
        iou_score=0.7421
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)

    # Generate PDF report
    req = GenerateReportRequest(doctor_notes="Khám đáy mắt thấy xuất tiết cứng và vi phình mạch rải rác.")
    rep_out = create_report(analysis_id=analysis.id, body=req, db=db_session, current_user=admin_user)
    
    assert rep_out.report_code.startswith("REP-")
    assert rep_out.pdf_path.endswith(".pdf")
    
    # Check physical file exists on disk
    from app.storage.service import storage_service
    abs_pdf = storage_service.get_absolute_path(rep_out.pdf_path)
    assert os.path.exists(abs_pdf), f"PDF report file must be written to disk at {abs_pdf}"
    assert os.path.getsize(abs_pdf) > 1000, "PDF report must have non-trivial file size"
