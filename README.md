# NỀN TẢNG PHÂN VÙNG VÀ ĐỊNH LƯỢNG TỔN THƯƠNG VÕNG MẠC BẰNG DEEP LEARNING

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2-blue.svg)](https://react.dev/)
[![Docker Compose](https://img.shields.io/badge/Docker-Multi--Container-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passed-brightgreen.svg)]()

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  
**Lĩnh vực:** Medical Image Processing & Clinical Decision Support Systems (CDSS)

---

## 1. TỔNG QUAN HỆ THỐNG

Dự án xây dựng một hệ thống y tế hoàn chỉnh, ứng dụng Deep Learning để phân tích ảnh đáy mắt màu (Color Fundus Photography), phân vùng ranh giới tổn thương ở cấp độ điểm ảnh và định lượng lâm sàng tự động:

- **Nhánh AI 1 (IDRiD):** Phân vùng đa tổn thương võng mạc tiểu đường gồm 5 lớp:
  - **MA:** Microaneurysms (Vi phình mao mạch) – Màu Đỏ
  - **HE:** Haemorrhages (Xuất huyết võng mạc) – Màu Cam
  - **EX:** Hard Exudates (Xuất tiết cứng) – Màu Vàng
  - **SE:** Soft Exudates (Xuất tiết mềm / Bông gòn) – Màu Xanh lá
  - **OD:** Optic Disc (Đĩa thị giác / Gai thị) – Màu Xanh dương
- **Nhánh AI 2 (DRIVE):** Phân vùng hệ thống cây mao mạch võng mạc trong trường thị giác FOV – Màu Tím.
- **Định lượng lâm sàng (Medical Evaluator):** Đếm số ổ tổn thương độc lập bằng giải thuật **Connected Components Analysis (8-connectivity)**, tính diện tích pixel và tỷ lệ phần trăm diện tích tổn thương trên vùng thị trường võng mạc hợp lệ (FOV %).
- **Bộ quan sát tương tác (HTML5 Canvas Medical Viewer):** Hỗ trợ Pan, Zoom mượt mà, trượt chia màn hình trước/sau (Split Slider), lồng ghép bán trong suốt (Opacity), bật/tắt từng lớp nhãn màu độc lập.
- **Kết xuất Phiếu kết quả xét nghiệm PDF:** Tự động tổng hợp dữ liệu, đối sánh ảnh gốc - mask và xuất file PDF chuẩn y khoa bằng **ReportLab**.

---

## 2. KIẾN TRÚC HỆ THỐNG & CÔNG NGHỆ

```
ReactJS Frontend (Vite + TypeScript + Tailwind CSS)
   │ (HTTP REST API / Bearer JWT)
   ▼
FastAPI Backend Gateway (Python 3.11 / 3.13)
   ├── PostgreSQL Database (Users, Patients, Analyses, Lesion Details, Reports)
   ├── Redis Message Broker & Cache
   ├── Celery Async Task Worker (Non-blocking AI Inference Pipeline)
   ├── Deep Learning Core (PyTorch: Baseline U-Net & Attention U-Net)
   ├── Medical Evaluator (OpenCV Connected Components & FOV normalizer)
   ├── Storage Engine (MinIO S3-compatible API + Local Disk Fallback)
   └── ReportLab PDF Clinical Document Generator
```

---

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```
DACN2/
├── backend/                  # Mã nguồn FastAPI Backend
│   ├── app/
│   │   ├── api/              # RESTful API Endpoints (auth, patients, images, analysis, reports, dashboard)
│   │   ├── core/             # Cấu hình hệ thống, bảo mật JWT, Cython compatibility hook
│   │   ├── database/         # Kết nối SQLAlchemy, Session, Base metadata
│   │   ├── models/           # ORM Database Models (User, Patient, Analysis, LesionDetail, Report)
│   │   ├── schemas/          # Pydantic V2 Request/Response Validation Schemas
│   │   ├── services/         # Inference Pipeline, ReportLab PDF Generator
│   │   ├── storage/          # Storage Service Adapter (MinIO + Local storage)
│   │   ├── workers/          # Celery App & Async Worker Tasks
│   │   └── main.py           # FastAPI Application Entrypoint & Startup Seeds
│   └── requirements.txt
│
├── ai/                       # Lõi Trí tuệ Nhân tạo & Deep Learning
│   ├── datasets/             # IDRiDDataset, DRIVEDataset, Data Loaders & Statistics Exporters
│   ├── models/               # PyTorch UNet & AttentionUNet (with Attention Gates)
│   ├── training/             # Loss functions (BCEDiceLoss), Synchronized Augmentations, train.py
│   ├── evaluation/           # Biomedical Metrics (Dice, IoU, Precision, Recall, Specificity) & Evaluator
│   ├── inference/            # RetinalPredictor Engine
│   ├── checkpoints/          # Model Weights (*_best_model.pth), Curves & Training Logs
│   └── config.py             # Hyperparameters, Image Size (512x512), Device Detection (CUDA/CPU)
│
├── frontend/                 # Giao diện người dùng ReactJS SPA
│   ├── src/
│   │   ├── components/       # CanvasViewer (HTML5 Canvas), Navbar, Sidebar
│   │   ├── pages/            # Dashboard, Patients, AnalysisStudio, Reports, Login
│   │   ├── services/         # Axios API client with JWT interceptor
│   │   ├── context/          # AuthContext (Session state)
│   │   ├── App.tsx           # React Router with Protected Routes
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── data/                     # Dữ liệu ảnh thực tế
│   ├── IDRiD/                # 81 ảnh IDRiD gốc và Ground Truth 5 lớp tổn thương (.tif)
│   ├── DRIVE/                # 40 ảnh DRIVE gốc, Ground Truth 1st_manual (.gif), FOV mask (.gif)
│   ├── idrid_statistics.csv  # Bảng thống kê phân bố tổn thương IDRiD thực tế
│   └── drive_statistics.csv  # Bảng thống kê mật độ mao mạch DRIVE thực tế
│
├── notebooks/                # Jupyter Notebooks nghiên cứu & thực nghiệm
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_evaluation.ipynb
│
├── docs/                     # Tài liệu báo cáo Đồ án tốt nghiệp 5 chương & Slide bảo vệ
│   ├── 01_tong_quan_de_tai.md
│   ├── 02_co_so_ly_thuyet.md
│   ├── 03_phan_tich_thiet_ke_he_thong.md
│   ├── 04_xay_dung_va_thuc_nghiem.md
│   ├── 05_ket_luan_va_huong_phat_trien.md
│   ├── SLIDES_DEFENSE.md     # Bộ Slide thuyết trình 15 trang bằng tiếng Anh
│   ├── DEFENSE_SCRIPT.md     # Kịch bản thuyết trình bảo vệ 10-15 phút
│   └── VIVA_QA_DEFENSE.md    # Bộ câu hỏi và trả lời phản biện chuyên sâu
│
├── tests/                    # Bộ kiểm thử tự động Pytest (14/14 Tests Passed)
├── Dockerfile.backend        # Docker image cho Backend & AI Worker
├── Dockerfile.frontend       # Docker multi-stage build cho React & Nginx
├── docker-compose.yml        # Điều phối 6 dịch vụ Docker container
├── ROADMAP.md                # Lộ trình kỹ thuật 4 giai đoạn
├── TEST_CASES.md             # Bảng đặc tả và kết quả kiểm thử chi tiết
└── README.md
```

---

## 4. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### Cách 1: Khởi chạy bằng Docker Compose (Khuyên dùng cho triển khai)
Chỉ cần một câu lệnh duy nhất để khởi động toàn bộ 6 dịch vụ (Frontend, Backend, AI Worker, Redis, PostgreSQL, MinIO):

```bash
docker compose up --build
```

- **Frontend Web UI:** Truy cập tại `http://localhost:5173`
- **Backend API & Swagger Docs:** Truy cập tại `http://localhost:8000/docs`
- **MinIO Console:** Truy cập tại `http://localhost:9001` (user: `minioadmin`, pass: `minioadmin123`)

---

### Cách 2: Chạy trực tiếp trên máy trạm cá nhân (Local Standalone)

#### 1. Yêu cầu môi trường
- Python 3.11 hoặc 3.13
- Node.js v18+ & npm
- Hỗ trợ cả CPU lẫn GPU NVIDIA (CUDA 12+)

#### 2. Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```

#### 3. Khởi chạy Backend API
```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*(Hệ thống sẽ tự động khởi tạo database SQLite `retina_medical.db` và nạp sẵn tài khoản Quản trị viên mẫu)*.

#### 4. Khởi chạy Frontend React
Mở một terminal mới:
```bash
cd frontend
npm install
npm run dev
```
Truy cập trình duyệt tại `http://localhost:5173`.

---

## 5. TÀI KHOẢN ĐĂNG NHẬP KIỂM THỬ (DEMO CREDENTIALS)

Hệ thống được khởi tạo sẵn tài khoản quản trị y tế để kiểm thử nhanh (có sẵn nút bấm 1-click tại trang Login):
- **Tài khoản:** `admin`
- **Mật khẩu:** `admin123`
- **Vai trò:** Administrator / Senior Doctor

---

## 6. HUẤN LUYỆN MÔ HÌNH DEEP LEARNING (TRAINING PIPELINE)

Hệ thống cung cấp script huấn luyện linh hoạt, hỗ trợ cấu hình đa dạng:

```bash
# Huấn luyện Baseline U-Net trên IDRiD (GPU CUDA)
python ai/training/train.py --dataset idrid --model unet --epochs 15 --batch_size 4 --device cuda

# Huấn luyện Attention U-Net trên IDRiD
python ai/training/train.py --dataset idrid --model attention_unet --epochs 15 --batch_size 4 --device cuda

# Huấn luyện phân vùng mạch máu trên DRIVE
python ai/training/train.py --dataset drive --model unet --epochs 15 --batch_size 4 --device cuda
```

Kết quả huấn luyện tự động lưu:
- Điểm kiểm tra trọng số tốt nhất: `ai/checkpoints/*_best_model.pth`
- Lịch sử số liệu thực nghiệm: `ai/checkpoints/*_training_history.csv`
- Đồ thị trực quan: `*_loss_curve.png`, `*_dice_curve.png`, `*_iou_curve.png`.

---

## 7. KIỂM THỬ HỆ THỐNG (AUTOMATED QUALITY ASSURANCE)

Hệ thống tích hợp bộ kiểm thử tự động toàn diện bằng Pytest:
```bash
python -m pytest tests/ -v
```

**Kết quả kiểm thử thực tế:**
```
tests/test_ai_pipeline.py::test_unet_architecture_forward PASSED         [  7%]
tests/test_ai_pipeline.py::test_attention_unet_forward PASSED            [ 14%]
tests/test_ai_pipeline.py::test_loss_functions PASSED                    [ 21%]
tests/test_ai_pipeline.py::test_metric_calculations PASSED               [ 28%]
tests/test_ai_pipeline.py::test_idrid_dataset_leakage_and_shapes PASSED  [ 35%]
tests/test_ai_pipeline.py::test_drive_dataset_shapes PASSED              [ 42%]
tests/test_backend_api.py::test_auth_registration_and_login PASSED       [ 50%]
tests/test_backend_api.py::test_patient_management_lifecycle PASSED      [ 57%]
tests/test_backend_api.py::test_image_samples_and_load PASSED            [ 64%]
tests/test_backend_api.py::test_dashboard_statistics_and_distributions PASSED [ 71%]
tests/test_backend_api.py::test_report_generation PASSED                 [ 78%]
tests/test_evaluator.py::test_extract_retinal_fov PASSED                 [ 85%]
tests/test_evaluator.py::test_analyze_connected_components PASSED        [ 92%]
tests/test_evaluator.py::test_render_color_mask_overlay PASSED           [100%]
====================== 14 passed in 12.83s =======================
```
*(Chi tiết từng ca kiểm thử được lưu trữ tại [TEST_CASES.md](TEST_CASES.md))*.

---

## 8. CẢNH BÁO Y ĐỨC VÀ PHÁP LÝ (MEDICAL DISCLAIMER)

Hệ thống phần mềm được thiết kế và phát triển với tư cách là một **Nền tảng hỗ trợ nghiên cứu khoa học và trợ giúp quyết định lâm sàng (Clinical Decision Support System - CDSS)**. Các đường biên phân vùng, số lượng ổ tổn thương và tỷ lệ phần trăm diện tích do mô hình Deep Learning tính toán không thay thế cho chẩn đoán xác định của bác sĩ chuyên khoa mắt có chứng chỉ hành nghề y khoa.
