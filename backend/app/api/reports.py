import os
import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database.session import get_db
from app.models.analysis import Analysis, Report, AnalysisStatus
from app.models.patient import Patient
from app.models.user import User
from app.schemas.report import ReportOut
from app.api.deps import get_current_user
from app.services.pdf_generator import generate_medical_pdf_report
from app.services.excel_generator import generate_medical_excel_report, generate_patients_summary_excel
from app.storage.service import storage_service

router = APIRouter(prefix="/reports", tags=["PDF & Excel Medical Reports"])

class GenerateReportRequest(BaseModel):
    doctor_notes: str = ""

@router.post("/generate/{analysis_id}", response_model=ReportOut)
def create_report(
    analysis_id: int,
    body: GenerateReportRequest = GenerateReportRequest(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Generates an official clinical PDF report for a completed analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    if analysis.analysis_status != AnalysisStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot generate report for analysis in status: {analysis.analysis_status}"
        )

    patient = db.query(Patient).filter(Patient.id == analysis.patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated patient not found")

    # Generate unique report code
    report_code = f"REP-{analysis.id}-{uuid.uuid4().hex[:6].upper()}"
    pdf_filename = f"{report_code}.pdf"
    target_dir = os.path.join(storage_service.local_root, "reports")
    os.makedirs(target_dir, exist_ok=True)
    pdf_abs_path = os.path.join(target_dir, pdf_filename)

    # Resolve image paths
    orig_abs = storage_service.get_absolute_path(analysis.original_image_url)
    pred_abs = storage_service.get_absolute_path(analysis.predicted_mask_url) if analysis.predicted_mask_url else None

    # Doctor name
    doctor_name = current_user.full_name or current_user.username

    # Generate PDF
    generate_medical_pdf_report(
        analysis=analysis,
        patient=patient,
        doctor_name=doctor_name,
        output_pdf_path=pdf_abs_path,
        original_img_path=orig_abs,
        predicted_mask_path=pred_abs,
        doctor_notes=body.doctor_notes
    )

    pdf_rel_url = f"/api/images/serve/reports/{pdf_filename}"

    # Check if report already exists for this analysis
    report = db.query(Report).filter(Report.analysis_id == analysis_id).first()
    if report:
        report.report_code = report_code
        report.pdf_path = pdf_rel_url
        report.doctor_notes = body.doctor_notes
    else:
        report = Report(
            analysis_id=analysis_id,
            report_code=report_code,
            pdf_path=pdf_rel_url,
            doctor_notes=body.doctor_notes
        )
        db.add(report)

    db.commit()
    db.refresh(report)
    return report

@router.get("/{id}/download")
def download_report(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download the clinical PDF report directly as an attachment."""
    report = db.query(Report).filter(Report.id == id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    abs_path = storage_service.get_absolute_path(report.pdf_path)
    if not abs_path or not os.path.exists(abs_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF report file not found on disk")

    return FileResponse(
        path=abs_path,
        media_type="application/pdf",
        filename=f"{report.report_code}.pdf",
        headers={"Content-Disposition": f"attachment; filename={report.report_code}.pdf"}
    )


@router.post("/export-excel/{analysis_id}")
def export_analysis_excel(
    analysis_id: int,
    body: GenerateReportRequest = GenerateReportRequest(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generates and downloads an Excel (.xlsx) clinical report for a completed analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    if analysis.analysis_status != AnalysisStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot export Excel for analysis in status: {analysis.analysis_status}"
        )

    patient = db.query(Patient).filter(Patient.id == analysis.patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated patient not found")

    report_code = f"REP-{analysis.id}-{uuid.uuid4().hex[:6].upper()}"
    xlsx_filename = f"{report_code}.xlsx"
    target_dir = os.path.join(storage_service.local_root, "reports")
    os.makedirs(target_dir, exist_ok=True)
    xlsx_abs_path = os.path.join(target_dir, xlsx_filename)

    doctor_name = current_user.full_name or current_user.username
    generate_medical_excel_report(
        analysis=analysis,
        patient=patient,
        doctor_name=doctor_name,
        output_xlsx_path=xlsx_abs_path,
        doctor_notes=body.doctor_notes,
    )

    return FileResponse(
        path=xlsx_abs_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=xlsx_filename,
        headers={
            "Content-Disposition": f"attachment; filename={xlsx_filename}",
            "X-Report-Code": report_code,
        }
    )


@router.get("/export-all-excel")
def export_all_patients_excel(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Exports all patients, diagnostic analyses, and lesion details to a multi-sheet Excel file."""
    patients = db.query(Patient).order_by(Patient.id.asc()).all()
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).all()

    xlsx_filename = "RetinaAI_Patients_Diagnoses.xlsx"
    target_dir = os.path.join(storage_service.local_root, "reports")
    os.makedirs(target_dir, exist_ok=True)
    xlsx_abs_path = os.path.join(target_dir, xlsx_filename)

    generate_patients_summary_excel(
        patients=patients,
        analyses=analyses,
        output_xlsx_path=xlsx_abs_path,
    )

    return FileResponse(
        path=xlsx_abs_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=xlsx_filename,
        headers={"Content-Disposition": f"attachment; filename={xlsx_filename}"}
    )

