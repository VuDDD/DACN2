import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from app.core.config import settings

def generate_medical_pdf_report(
    analysis: Any,
    patient: Any,
    doctor_name: str,
    output_pdf_path: str,
    original_img_path: Optional[str] = None,
    predicted_mask_path: Optional[str] = None,
    doctor_notes: Optional[str] = None,
) -> str:
    """
    Generates a professional clinical PDF report for retinal lesion and vessel segmentation.
    Returns: output_pdf_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0284c7") # Medical Sky Blue
    navy_color = colors.HexColor("#0f172a") # Slate Navy
    secondary_color = colors.HexColor("#0d9488") # Medical Teal
    gray_bg = colors.HexColor("#f8fafc")
    border_color = colors.HexColor("#e2e8f0")

    title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=navy_color,
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "HeaderSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=primary_color,
        alignment=TA_CENTER
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=navy_color
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=navy_color,
        alignment=TA_CENTER
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_JUSTIFY
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("VIETNAM - KOREA UNIVERSITY OF INFORMATION & COMMUNICATION TECHNOLOGY (VKU)", subtitle_style))
    story.append(Paragraph("RETINA AI CLINICAL QUANTIFICATION REPORT", title_style))
    story.append(Paragraph("Automated Deep Learning Retinal Lesion & Microvascular Segmentation Platform", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=12))

    # 2. Patient & Exam Info Table
    dob_str = str(patient.date_of_birth) if getattr(patient, "date_of_birth", None) else "N/A"
    exam_date_str = analysis.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(analysis.created_at, "strftime") else str(analysis.created_at)

    info_data = [
        [
            Paragraph("<b>Patient Name:</b> " + str(patient.full_name), body_style),
            Paragraph("<b>Patient Code:</b> " + str(patient.patient_code), body_style),
            Paragraph("<b>Exam Date:</b> " + exam_date_str, body_style),
        ],
        [
            Paragraph("<b>Gender:</b> " + str(patient.gender), body_style),
            Paragraph("<b>Date of Birth:</b> " + dob_str, body_style),
            Paragraph("<b>Physician:</b> Dr. " + str(doctor_name), body_style),
        ],
        [
            Paragraph("<b>Analysis ID:</b> #" + str(analysis.id), body_style),
            Paragraph("<b>Task:</b> " + str(analysis.task_type), body_style),
            Paragraph("<b>AI Model:</b> " + str(analysis.model_name), body_style),
        ]
    ]

    info_table = Table(info_data, colWidths=[180, 170, 170])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), gray_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 14))

    # 3. Medical Imaging Section (Side-by-side or stacked images)
    story.append(Paragraph("1. Retinal Imaging & AI Segmentation Overlays", section_title_style))
    
    img_cells = []
    headers = []
    img_w, img_h = 240, 180

    if original_img_path and os.path.exists(original_img_path):
        try:
            rl_orig = RLImage(original_img_path, width=img_w, height=img_h)
            headers.append(Paragraph("<b>Original Retinal Photograph</b>", table_cell_style))
            img_cells.append(rl_orig)
        except Exception:
            pass

    if predicted_mask_path and os.path.exists(predicted_mask_path):
        try:
            rl_pred = RLImage(predicted_mask_path, width=img_w, height=img_h)
            headers.append(Paragraph("<b>AI Predicted Lesion / Vessel Overlay</b>", table_cell_style))
            img_cells.append(rl_pred)
        except Exception:
            pass

    if img_cells:
        if len(img_cells) == 2:
            img_table_data = [headers, img_cells]
            img_table = Table(img_table_data, colWidths=[260, 260])
        else:
            img_table_data = [headers, img_cells]
            img_table = Table(img_table_data, colWidths=[520])

        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(img_table)
    else:
        story.append(Paragraph("<i>[Image preview visual unavailable in offline rendering]</i>", body_style))

    story.append(Spacer(1, 14))

    # 4. Quantitative Lesion & Biomarker Findings Table
    story.append(Paragraph("2. Quantitative Medical Findings & Area Measurement", section_title_style))

    lesion_headers = [
        Paragraph("Lesion / Structure", table_header_style),
        Paragraph("Foci Count", table_header_style),
        Paragraph("Abnormal Area (px)", table_header_style),
        Paragraph("Retinal FOV Ratio (%)", table_header_style),
        Paragraph("Clinical Significance", table_header_style),
    ]

    lesion_rows = [lesion_headers]
    lesion_names_map = {
        "MA": ("Microaneurysms (MA)", "Early indicator of diabetic retinopathy micro-vascular compromise"),
        "HE": ("Haemorrhages (HE)", "Ruptured retinal capillary walls, signs of severe non-proliferative DR"),
        "EX": ("Hard Exudates (EX)", "Lipid and lipoprotein leakage from leaking retinal capillaries"),
        "SE": ("Soft Exudates / Cotton Wool (SE)", "Micro-infarction of nerve fiber layer resulting in focal ischemia"),
        "OD": ("Optic Disc (OD)", "Normal anatomical structure / anatomical landmark"),
        "VESSEL": ("Retinal Blood Vessels", "Full microvascular tree within valid Field of View (FOV)"),
    }

    details = getattr(analysis, "lesion_details", [])
    if details:
        for d in details:
            info = lesion_names_map.get(d.lesion_type, (d.lesion_type, "Abnormal retinal manifestation"))
            row = [
                Paragraph(f"<b>{info[0]}</b>", table_cell_style),
                Paragraph(str(d.count), table_cell_style),
                Paragraph(f"{d.area_pixels:,} px", table_cell_style),
                Paragraph(f"{d.area_percentage:.3f} %", table_cell_style),
                Paragraph(f"<font size='7.5'>{info[1]}</font>", table_cell_style),
            ]
            lesion_rows.append(row)
    else:
        # Default row if no detailed breakdown is registered
        total_px = getattr(analysis, "lesion_area_pixels", 0) or 0
        total_pct = getattr(analysis, "lesion_percentage", 0.0) or 0.0
        lesion_rows.append([
            Paragraph("Total Abnormal Lesions", table_cell_style),
            Paragraph("Detected", table_cell_style),
            Paragraph(f"{total_px:,} px", table_cell_style),
            Paragraph(f"{total_pct:.3f} %", table_cell_style),
            Paragraph("Aggregate pathological presentation", table_cell_style),
        ])

    lesion_table = Table(lesion_rows, colWidths=[120, 60, 95, 95, 150])
    lesion_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, gray_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(lesion_table)
    story.append(Spacer(1, 14))

    # 5. Model Evaluation Metrics (if ground truth was provided)
    if analysis.dice_score is not None:
        story.append(Paragraph("3. Model Validation Metrics (Evaluated vs Ground Truth)", section_title_style))
        metric_headers = [
            Paragraph("Dice Score", table_header_style),
            Paragraph("IoU (Jaccard)", table_header_style),
            Paragraph("Precision", table_header_style),
            Paragraph("Recall / Sensitivity", table_header_style),
            Paragraph("Specificity", table_header_style),
            Paragraph("F1-Score", table_header_style),
        ]
        metric_values = [
            Paragraph(f"{analysis.dice_score:.4f}" if analysis.dice_score else "N/A", table_cell_style),
            Paragraph(f"{analysis.iou_score:.4f}" if analysis.iou_score else "N/A", table_cell_style),
            Paragraph(f"{analysis.precision:.4f}" if analysis.precision else "N/A", table_cell_style),
            Paragraph(f"{analysis.recall:.4f}" if analysis.recall else "N/A", table_cell_style),
            Paragraph(f"{analysis.specificity:.4f}" if analysis.specificity else "N/A", table_cell_style),
            Paragraph(f"{analysis.f1_score:.4f}" if analysis.f1_score else "N/A", table_cell_style),
        ]
        metric_table = Table([metric_headers, metric_values], colWidths=[86, 86, 86, 86, 86, 86])
        metric_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), secondary_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(metric_table)
        story.append(Spacer(1, 14))

    # 6. Physician Notes & Signature
    notes_content = doctor_notes or getattr(analysis, "error_message", None) or "Automated screening performed. Lesion boundaries identified and quantified."
    story.append(Paragraph("4. Physician Observations & Clinical Notes", section_title_style))
    notes_p = Paragraph(f"<i>{notes_content}</i>", body_style)
    notes_table = Table([[notes_p]], colWidths=[520])
    notes_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), gray_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(notes_table)
    story.append(Spacer(1, 20))

    # Signature Block
    sig_data = [
        [
            Paragraph("<b>System Operator / AI Engine:</b><br/>Retina AI v1.0 (PyTorch Core)", body_style),
            Paragraph("<b>Consulting Physician Signature:</b><br/><br/>____________________________________<br/>Dr. " + str(doctor_name), body_style),
        ]
    ]
    sig_table = Table(sig_data, colWidths=[260, 260])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 15))

    # 7. Mandatory Legal Disclaimer
    disclaimer_text = (
        "<b>CLINICAL RESEARCH & DECISION SUPPORT DISCLAIMER:</b> This report is generated by an artificial intelligence "
        "system (Deep Learning Convolutional Neural Network) developed for research and clinical assistance. "
        "The quantification metrics, lesion segmentation boundaries, and vessel classifications provided herein are intended "
        "solely to assist medical professionals and do not constitute a definitive medical diagnosis or replacement for "
        "clinical evaluation by a board-certified ophthalmologist. Quantitative area measurements are provided in discrete "
        "pixel counts and Field of View (FOV) percentages unless optical spatial calibration data is explicitly verified."
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build Document
    doc.build(story)
    return output_pdf_path

