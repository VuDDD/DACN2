import os
from typing import Any, List, Optional
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from ai.evaluation.evaluator import classify_npdr_severity

LESION_META = {
    "MA": ("Microaneurysms (MA)", "Dấu hiệu sớm của thoái hóa vi mạch võng mạc đái tháo đường"),
    "HE": ("Haemorrhages (HE)", "Vỡ thành mao mạch võng mạc, dấu hiệu bệnh võng mạc tiến triển"),
    "EX": ("Hard Exudates (EX)", "Lắng đọng lipid và lipoprotein rò rỉ từ mao mạch tổn thương"),
    "SE": ("Soft Exudates / Cotton Wool (SE)", "Nhồi máu lớp sợi thần kinh võng mạc do thiếu máu cục bộ"),
    "OD": ("Optic Disc (OD)", "Cấu trúc giải phẫu bình thường (Đĩa thị giác)"),
    "VESSEL": ("Retinal Blood Vessels", "Hệ thống cây mao mạch võng mạc trong vùng FOV hợp lệ"),
}

GRADE_FILL_HEX = {
    "No DR": "D1FAE5",
    "Mild": "DBEAFE",
    "Moderate": "FEF3C7",
    "Severe": "FEE2E2",
}

THIN_BORDER = Border(
    left=Side(style="thin", color="CBD5E1"),
    right=Side(style="thin", color="CBD5E1"),
    top=Side(style="thin", color="CBD5E1"),
    bottom=Side(style="thin", color="CBD5E1"),
)


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert numeric or legacy SQLite little-endian int32/int64 bytes to Python int."""
    if val is None:
        return default
    if isinstance(val, (bytes, bytearray)):
        return int.from_bytes(val, byteorder="little", signed=True)
    return int(val)


def _auto_fit_columns(ws, min_width: int = 12, max_width: int = 48) -> None:
    """Adjust worksheet column widths based on cell string lengths."""
    for col_idx in range(1, ws.max_column + 1):
        max_len = 0
        for row_idx in range(1, ws.max_row + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                for line in str(val).splitlines():
                    max_len = max(max_len, len(line))
        ws.column_dimensions[get_column_letter(col_idx)].width = max(min_width, min(max_len + 4, max_width))


def _style_header_row(ws, row_idx: int, col_count: int, fill_hex: str = "0284C7") -> None:
    fill = PatternFill("solid", fgColor=fill_hex)
    font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    for c in range(1, col_count + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER


def generate_medical_excel_report(
    analysis: Any,
    patient: Any,
    doctor_name: str,
    output_xlsx_path: str,
    doctor_notes: Optional[str] = None,
) -> str:
    """
    Generates a clinical Excel (.xlsx) report matching UC-05 structure:
    - Sheet 1 (Phieu Ket Qua): VKU header, patient info, 6-column quantification table,
      ICDR NPDR staging, AI evaluation metrics, doctor notes & legal disclaimer.
    - Sheet 2 (Chi Tiet Du Lieu): Normalized tabular lesion breakdown for data analysis.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_xlsx_path)), exist_ok=True)
    wb = Workbook()

    ws = wb.active
    ws.title = "Phieu Ket Qua"

    # Banner Header
    ws.merge_cells("A1:F1")
    ws["A1"] = "VIETNAM - KOREA UNIVERSITY OF INFORMATION & COMMUNICATION TECHNOLOGY (VKU)"
    ws["A1"].font = Font(name="Arial", size=10, bold=True, color="0284C7")
    ws["A1"].alignment = Alignment(horizontal="center")

    ws.merge_cells("A2:F2")
    ws["A2"] = "PHIẾU KẾT QUẢ ĐỊNH LƯỢNG TỔN THƯƠNG VÕNG MẠC (RETINA AI CLINICAL REPORT)"
    ws["A2"].font = Font(name="Arial", size=13, bold=True, color="0F172A")
    ws["A2"].alignment = Alignment(horizontal="center")

    dob_str = str(patient.date_of_birth) if getattr(patient, "date_of_birth", None) else "N/A"
    exam_date = analysis.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(analysis.created_at, "strftime") else str(analysis.created_at)

    # Patient & Exam Information
    ws["A4"] = "1. THÔNG TIN HÀNH CHÍNH BỆNH NHÂN & CA CHẨN ĐOÁN"
    ws["A4"].font = Font(name="Arial", size=11, bold=True, color="0284C7")

    info_rows = [
        ("Họ và tên bệnh nhân", patient.full_name, "Mã bệnh nhân", patient.patient_code, "Ngày khám", exam_date),
        ("Giới tính", patient.gender, "Ngày sinh", dob_str, "Bác sĩ phụ trách", f"Dr. {doctor_name}"),
        ("Mã phân tích", f"#{analysis.id}", "Nhiệm vụ", analysis.task_type, "Mô hình AI", analysis.model_name),
        ("Tiền sử bệnh lý", patient.medical_history or "Không ghi nhận", "Số điện thoại", patient.phone_number or "N/A", "Diện tích FOV (px)", f"{analysis.retinal_fov_pixels or 0:,}"),
    ]
    label_fill = PatternFill("solid", fgColor="F1F5F9")
    for offset, row_vals in enumerate(info_rows, start=5):
        for col_i, val in enumerate(row_vals, start=1):
            cell = ws.cell(row=offset, column=col_i, value=val)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if col_i in (1, 3, 5):
                cell.font = Font(name="Arial", size=9.5, bold=True, color="334155")
                cell.fill = label_fill
            else:
                cell.font = Font(name="Arial", size=9.5, color="0F172A")

    # Quantitative Table (6 columns)
    row_cursor = 10
    ws.cell(row=row_cursor, column=1, value="2. BẢNG ĐỊNH LƯỢNG Y HỌC & CHỈ SỐ VÙNG TỔN THƯƠNG").font = Font(
        name="Arial", size=11, bold=True, color="0284C7"
    )
    row_cursor += 1
    headers = ["Tên tổn thương / Cấu trúc", "Số ổ (Foci)", "Diện tích (px)", "Diện tích (mm²)", "Tỷ lệ võng mạc (% FOV)", "Ý nghĩa lâm sàng"]
    for idx, h in enumerate(headers, start=1):
        ws.cell(row=row_cursor, column=idx, value=h)
    _style_header_row(ws, row_cursor, 6)

    retinal_fov_px = getattr(analysis, "retinal_fov_pixels", 0) or 0
    details = getattr(analysis, "lesion_details", []) or []
    lesion_dicts = []

    if details:
        for d in details:
            row_cursor += 1
            meta = LESION_META.get(d.lesion_type, (d.lesion_type, "Bất thường võng mạc"))
            d_cnt = _safe_int(d.count)
            d_area = _safe_int(d.area_pixels)
            d_pct = float(d.area_percentage or 0.0)
            area_mm2 = (d_area / retinal_fov_px) * 1090.0 if retinal_fov_px > 0 else (d_pct / 100.0) * 1090.0
            vals = [meta[0], d_cnt, d_area, round(area_mm2, 3), round(d_pct, 4), meta[1]]
            for c_idx, v in enumerate(vals, start=1):
                cell = ws.cell(row=row_cursor, column=c_idx, value=v)
                cell.border = THIN_BORDER
                cell.font = Font(name="Arial", size=9.5, bold=(c_idx == 1), color="0F172A")
                cell.alignment = Alignment(
                    horizontal="left" if c_idx in (1, 6) else "center",
                    vertical="center",
                    wrap_text=True,
                )
            lesion_dicts.append({
                "lesion_type": d.lesion_type,
                "count": d_cnt,
                "area_pixels": d_area,
                "area_percentage": d_pct,
            })
    else:
        row_cursor += 1
        total_px = getattr(analysis, "lesion_area_pixels", 0) or 0
        total_pct = getattr(analysis, "lesion_percentage", 0.0) or 0.0
        total_mm2 = (total_px / retinal_fov_px) * 1090.0 if retinal_fov_px > 0 else (total_pct / 100.0) * 1090.0
        vals = ["Tổng tổn thương bất thường", 0, int(total_px), round(total_mm2, 3), round(float(total_pct), 4), "Tổng hợp vùng tổn thương"]
        for c_idx, v in enumerate(vals, start=1):
            cell = ws.cell(row=row_cursor, column=c_idx, value=v)
            cell.border = THIN_BORDER
            cell.font = Font(name="Arial", size=9.5, color="0F172A")

    # NPDR Staging Section
    row_cursor += 2
    ws.cell(row=row_cursor, column=1, value="3. PHÂN TẦNG MỨC ĐỘ NPDR THEO CHUẨN ICDR").font = Font(
        name="Arial", size=11, bold=True, color="0284C7"
    )
    total_pct_val = float(getattr(analysis, "lesion_percentage", 0.0) or 0.0)
    npdr_info = classify_npdr_severity(lesion_dicts, total_pct_val)
    grade_fill = PatternFill("solid", fgColor=GRADE_FILL_HEX.get(npdr_info["grade"], "E0F2FE"))

    row_cursor += 1
    npdr_rows = [
        ("Chẩn đoán phân tầng", f"{npdr_info['grade'].upper()} NPDR (ICDR Level {npdr_info.get('icdr_level', 0)})"),
        ("Mô tả bệnh học", npdr_info["description"]),
        ("Khuyến nghị lâm sàng", npdr_info["recommendation"]),
    ]
    for idx, (k, v) in enumerate(npdr_rows):
        r = row_cursor + idx
        c_key = ws.cell(row=r, column=1, value=k)
        c_key.font = Font(name="Arial", size=9.5, bold=True, color="0F172A")
        c_key.fill = grade_fill if idx == 0 else label_fill
        c_key.border = THIN_BORDER
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
        c_val = ws.cell(row=r, column=2, value=v)
        c_val.font = Font(name="Arial", size=9.5, bold=(idx == 0), color="0F172A")
        c_val.fill = grade_fill if idx == 0 else PatternFill(fill_type=None)
        c_val.alignment = Alignment(wrap_text=True, vertical="center")
        for col_b in range(2, 7):
            ws.cell(row=r, column=col_b).border = THIN_BORDER
    row_cursor += len(npdr_rows)

    # AI Evaluation Metrics (if Ground Truth available)
    if analysis.dice_score is not None:
        row_cursor += 1
        ws.cell(row=row_cursor, column=1, value="4. CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH AI (SO VỚI GROUND TRUTH)").font = Font(
            name="Arial", size=11, bold=True, color="0284C7"
        )
        row_cursor += 1
        m_headers = ["Dice Score", "IoU (Jaccard)", "Precision", "Recall / Sensitivity", "Specificity", "F1-Score"]
        m_vals = [
            round(analysis.dice_score or 0.0, 4),
            round(analysis.iou_score or 0.0, 4),
            round(analysis.precision or 0.0, 4),
            round(analysis.recall or 0.0, 4),
            round(analysis.specificity or 0.0, 4),
            round(analysis.f1_score or 0.0, 4),
        ]
        for idx, h in enumerate(m_headers, start=1):
            ws.cell(row=row_cursor, column=idx, value=h)
        _style_header_row(ws, row_cursor, 6, fill_hex="0D9488")
        row_cursor += 1
        for idx, v in enumerate(m_vals, start=1):
            cell = ws.cell(row=row_cursor, column=idx, value=v)
            cell.border = THIN_BORDER
            cell.font = Font(name="Arial", size=10, bold=True, color="0F172A")
            cell.alignment = Alignment(horizontal="center", vertical="center")

    # Physician Notes & Disclaimer
    row_cursor += 2
    ws.cell(row=row_cursor, column=1, value="5. GHI CHÚ LÂM SÀNG CỦA BÁC SĨ & XÁC NHẬN").font = Font(
        name="Arial", size=11, bold=True, color="0284C7"
    )
    row_cursor += 1
    notes_content = doctor_notes or getattr(analysis, "error_message", None) or "Đã sàng lọc tự động và định lượng vùng tổn thương võng mạc."
    ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=6)
    n_cell = ws.cell(row=row_cursor, column=1, value=notes_content)
    n_cell.font = Font(name="Arial", size=9.5, italic=True, color="0F172A")
    n_cell.alignment = Alignment(wrap_text=True, vertical="center")
    for col_b in range(1, 7):
        ws.cell(row=row_cursor, column=col_b).border = THIN_BORDER

    row_cursor += 1
    ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=3)
    ws.cell(row=row_cursor, column=1, value="Hệ thống vận hành: Retina AI v1.0 (PyTorch Core)").font = Font(name="Arial", size=9, bold=True)
    ws.merge_cells(start_row=row_cursor, start_column=4, end_row=row_cursor, end_column=6)
    ws.cell(row=row_cursor, column=4, value=f"Bác sĩ phụ trách: Dr. {doctor_name}").font = Font(name="Arial", size=9, bold=True)

    row_cursor += 2
    ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=6)
    d_cell = ws.cell(
        row=row_cursor,
        column=1,
        value="TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM: Kết quả định lượng AI chỉ mang tính hỗ trợ quyết định lâm sàng (CDSS), không thay thế chẩn đoán xác định của bác sĩ chuyên khoa mắt.",
    )
    d_cell.font = Font(name="Arial", size=8, italic=True, color="64748B")
    d_cell.alignment = Alignment(wrap_text=True)

    _auto_fit_columns(ws, min_width=16, max_width=45)

    # Sheet 2: Raw Breakdown Data
    ws2 = wb.create_sheet("Chi Tiet Du Lieu")
    s2_headers = ["Ma Phan Tich", "Ma Benh Nhan", "Ho Ten", "Nhiem Vu", "Mo Hinh AI", "Loai Ton Thuong", "So O (Foci)", "Dien Tich (px)", "Dien Tich (mm2)", "Ty Le FOV (%)"]
    for idx, h in enumerate(s2_headers, start=1):
        ws2.cell(row=1, column=idx, value=h)
    _style_header_row(ws2, 1, len(s2_headers))

    for r_i, d in enumerate(details, start=2):
        d_cnt = _safe_int(d.count)
        d_area = _safe_int(d.area_pixels)
        d_pct = float(d.area_percentage or 0.0)
        area_mm2 = (d_area / retinal_fov_px) * 1090.0 if retinal_fov_px > 0 else (d_pct / 100.0) * 1090.0
        row_data = [
            analysis.id,
            patient.patient_code,
            patient.full_name,
            analysis.task_type,
            analysis.model_name,
            d.lesion_type,
            d_cnt,
            d_area,
            round(area_mm2, 4),
            round(d_pct, 4),
        ]
        for c_i, val in enumerate(row_data, start=1):
            c = ws2.cell(row=r_i, column=c_i, value=val)
            c.border = THIN_BORDER
            c.font = Font(name="Arial", size=9.5)
    _auto_fit_columns(ws2, min_width=14, max_width=32)

    wb.save(output_xlsx_path)
    return output_xlsx_path


def generate_patients_summary_excel(
    patients: List[Any],
    analyses: List[Any],
    output_xlsx_path: str,
) -> str:
    """
    Exports full clinical registry of patients, diagnostic analyses, and lesion details
    into a 3-sheet Excel workbook.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_xlsx_path)), exist_ok=True)
    wb = Workbook()

    # Sheet 1: Patients
    ws_p = wb.active
    ws_p.title = "Danh Sach Benh Nhan"
    p_headers = ["ID", "Mã Bệnh Nhân", "Họ và Tên", "Giới Tính", "Ngày Sinh", "Số Điện Thoại", "Tiền Sử Bệnh Lý", "Ghi Chú", "Số Ca Phân Tích"]
    for idx, h in enumerate(p_headers, start=1):
        ws_p.cell(row=1, column=idx, value=h)
    _style_header_row(ws_p, 1, len(p_headers))

    patient_map = {}
    for r_i, p in enumerate(patients, start=2):
        patient_map[p.id] = p
        row_vals = [
            p.id,
            p.patient_code,
            p.full_name,
            p.gender,
            str(p.date_of_birth) if p.date_of_birth else "",
            p.phone_number or "",
            p.medical_history or "",
            p.notes or "",
            len(p.analyses),
        ]
        for c_i, val in enumerate(row_vals, start=1):
            cell = ws_p.cell(row=r_i, column=c_i, value=val)
            cell.border = THIN_BORDER
            cell.font = Font(name="Arial", size=9.5)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    _auto_fit_columns(ws_p, min_width=12, max_width=42)

    # Sheet 2: Analyses
    ws_a = wb.create_sheet("Ket Qua Chan Doan")
    a_headers = [
        "Mã Xét Nghiệm", "Mã Bệnh Nhân", "Họ Tên", "Nhiệm Vụ", "Mô Hình AI", "Trạng Thái",
        "FOV (px)", "Tổn Thương (px)", "Tỷ Lệ (% FOV)", "Phân Tầng ICDR",
        "Dice", "IoU", "Precision", "Recall", "Thời Gian (ms)", "Ngày Tạo"
    ]
    for idx, h in enumerate(a_headers, start=1):
        ws_a.cell(row=1, column=idx, value=h)
    _style_header_row(ws_a, 1, len(a_headers), fill_hex="0D9488")

    # Sheet 3: Lesion Details
    ws_l = wb.create_sheet("Chi Tiet Ton Thuong")
    l_headers = ["Mã Xét Nghiệm", "Mã Bệnh Nhân", "Họ Tên", "Loại Tổn Thương", "Số Ổ (Foci)", "Diện Tích (px)", "Diện Tích (mm²)", "Tỷ Lệ (% FOV)"]
    for idx, h in enumerate(l_headers, start=1):
        ws_l.cell(row=1, column=idx, value=h)
    _style_header_row(ws_l, 1, len(l_headers), fill_hex="334155")

    l_row = 2
    for r_i, a in enumerate(analyses, start=2):
        p = patient_map.get(a.patient_id)
        p_code = p.patient_code if p else f"ID-{a.patient_id}"
        p_name = p.full_name if p else ""
        details = getattr(a, "lesion_details", []) or []
        lesion_dicts = [
            {
                "lesion_type": d.lesion_type,
                "count": _safe_int(d.count),
                "area_pixels": _safe_int(d.area_pixels),
                "area_percentage": float(d.area_percentage or 0.0),
            }
            for d in details
        ]
        npdr = classify_npdr_severity(lesion_dicts, float(a.lesion_percentage or 0.0)) if a.task_type == "IDRID_LESIONS" else {"grade": "Vessel Mapping"}
        created_str = a.created_at.strftime("%Y-%m-%d %H:%M") if hasattr(a.created_at, "strftime") else str(a.created_at)

        fov_px = _safe_int(a.retinal_fov_pixels)
        total_px = _safe_int(a.lesion_area_pixels)
        a_vals = [
            a.id, p_code, p_name, a.task_type, a.model_name, a.analysis_status,
            fov_px, total_px, round(float(a.lesion_percentage or 0.0), 4),
            npdr["grade"],
            round(a.dice_score, 4) if a.dice_score is not None else "",
            round(a.iou_score, 4) if a.iou_score is not None else "",
            round(a.precision, 4) if a.precision is not None else "",
            round(a.recall, 4) if a.recall is not None else "",
            _safe_int(a.processing_time_ms), created_str,
        ]
        for c_i, val in enumerate(a_vals, start=1):
            cell = ws_a.cell(row=r_i, column=c_i, value=val)
            cell.border = THIN_BORDER
            cell.font = Font(name="Arial", size=9.5)

        for d in details:
            d_cnt = _safe_int(d.count)
            d_area = _safe_int(d.area_pixels)
            d_pct = float(d.area_percentage or 0.0)
            area_mm2 = (d_area / fov_px) * 1090.0 if fov_px > 0 else (d_pct / 100.0) * 1090.0
            l_vals = [a.id, p_code, p_name, d.lesion_type, d_cnt, d_area, round(area_mm2, 4), round(d_pct, 4)]
            for c_i, val in enumerate(l_vals, start=1):
                cell = ws_l.cell(row=l_row, column=c_i, value=val)
                cell.border = THIN_BORDER
                cell.font = Font(name="Arial", size=9.5)
            l_row += 1

    _auto_fit_columns(ws_a, min_width=12, max_width=32)
    _auto_fit_columns(ws_l, min_width=12, max_width=32)

    wb.save(output_xlsx_path)
    return output_xlsx_path
