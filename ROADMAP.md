# ROADMAP DỰ ÁN: NỀN TẢNG PHÂN VÙNG VÀ ĐỊNH LƯỢNG TỔN THƯƠNG VÕNG MẠC BẰNG DEEP LEARNING

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Lĩnh vực:** Medical Image Processing & Deep Learning Platform  

---

## 1. TỔNG QUAN VÀ MỤC TIÊU HỆ THỐNG

Dự án hướng đến việc xây dựng một hệ thống y tế hỗ trợ bác sĩ nhãn khoa và chuyên gia chẩn đoán hình ảnh:
- **Tự động phân vùng tổn thương vi mô (Microaneurysms - MA, Haemorrhages - HE, Hard Exudates - EX, Soft Exudates - SE) và giải phẫu đĩa thị (Optic Disc - OD)** trên ảnh đáy mắt võng mạc tiểu đường (Diabetic Retinopathy) dựa trên bộ dữ liệu chuẩn **IDRiD**.
- **Tự động phân vùng hệ thống mạch máu võng mạc (Blood Vessels)** trong vùng trường thị giác (Field of View - FOV) dựa trên bộ dữ liệu chuẩn **DRIVE**.
- **Định lượng lâm sàng tự động (Medical Quantification):** Đếm số ổ tổn thương bằng thuật toán Connected Components, đo lường diện tích tổn thương chính xác theo pixel và tỷ lệ phần trăm trên diện tích võng mạc hợp lệ (FOV), cung cấp các chỉ số đánh giá y sinh (Dice, IoU, Precision, Recall, Specificity).
- **Nền tảng Full-stack hoàn chỉnh:** Giao diện trực quan hoá đa lớp bằng HTML5 Canvas, kiểm soát độ trong suốt (opacity), phóng to/thu nhỏ (pan & zoom), so sánh ảnh gốc - ground truth - dự đoán; API chuẩn hoá FastAPI, cơ chế xử lý hàng đợi bất đồng bộ Celery + Redis, lưu trữ đối tượng MinIO S3, cơ sở dữ liệu PostgreSQL, và kết xuất phiếu kết quả xét nghiệm y khoa dạng PDF bằng ReportLab.

---

## 2. KIẾN TRÚC TỔNG THỂ (SYSTEM ARCHITECTURE)

```
+-----------------------------------------------------------------------------------+
|                              REACTJS FRONTEND (Vite + TS)                         |
|  - Auth & Patient Management       - HTML5 Canvas Medical Viewer (Pan/Zoom/Layers)|
|  - Image Upload & Preview          - Quantitative Clinical Dashboard (Charts)     |
|  - Real-time Polling / Status      - PDF Report Download                          |
+------------------------------------------+----------------------------------------+
                                           | HTTP / REST API (JWT Auth)
                                           v
+-----------------------------------------------------------------------------------+
|                               FASTAPI BACKEND SERVICE                             |
|  - /api/auth: Login, Register, Current User (Role: Admin / Doctor)                |
|  - /api/patients: CRUD bệnh nhân & bệnh án                                        |
|  - /api/images: Upload, validation, metadata extraction                           |
|  - /api/analysis: Task dispatching, progress tracking, result retrieval           |
|  - /api/reports: PDF Report generator trigger & download                          |
|  - /api/dashboard: Statistical metrics & lesion distribution                      |
+---------------------+--------------------+--------------------+-------------------+
                      |                    |                    |
                      v                    v                    v
          +----------------------+ +---------------+ +----------------------+
          | PostgreSQL Database  | |  Redis Broker | | MinIO Object Storage |
          | - users              | |  - Celery Que | | - Raw Retinal Images |
          | - patients           | |  - Task State | | - Ground Truth Masks |
          | - analyses & lesions | +-------+-------+ | - Predicted Overlays |
          | - reports            |         |         +----------------------+
          +----------------------+         |
                                           v
+-----------------------------------------------------------------------------------+
|                         CELERY ASYNC WORKER / AI ENGINE                           |
|  - Preprocessing & CLAHE Enhancement                                              |
|  - Dual-Branch Inference:                                                         |
|      * Branch 1 (IDRiD): 5-channel Lesion Segmentation (MA, HE, EX, SE, OD)       |
|      * Branch 2 (DRIVE): Binary Vessel Segmentation within Valid FOV              |
|  - Models: Baseline U-Net & Attention U-Net (PyTorch)                            |
|  - Medical Evaluator: Connected Components Counting, Pixel Area & FOV Ratio      |
|  - ReportLab PDF Generator Engine                                                 |
+-----------------------------------------------------------------------------------+
```

---

## 3. LỘ TRÌNH TRIỂN KHAI 4 GIAI ĐOẠN (ROADMAP PHASES)

### Giai đoạn 1: Khởi tạo Nền tảng, Database & AI Baseline
- [x] Khởi tạo Git repository và cấu hình `.gitignore` chuẩn y sinh/deep learning.
- [x] Xây dựng `ROADMAP.md` chi tiết.
- [ ] Thiết lập cấu trúc thư mục phân tách độc lập: `ai/`, `backend/`, `frontend/`, `data/`, `docs/`, `tests/`.
- [ ] Tổ chức dữ liệu từ tập ảnh thực tế sang `data/IDRiD` và `data/DRIVE`.
- [ ] Thiết kế cơ sở dữ liệu (Database Schema, SQLAlchemy ORM, Pydantic V2 schemas).
- [ ] Xây dựng Backend FastAPI (JWT Auth, Patients CRUD, Image Upload, Analysis task endpoint, Dashboard stats).
- [ ] Xây dựng AI Dataset Pipeline (`idrid_dataset.py`, `drive_dataset.py`, Augmentations, DataLoaders, Data statistics exporter).
- [ ] Xây dựng kiến trúc mô hình **Baseline U-Net** (`unet.py`).
- [ ] Xây dựng Loss functions kết hợp (Dice Loss, BCE Loss, Combo Loss) & Evaluation Metrics (`metrics.py`).
- [ ] Xây dựng pipeline huấn luyện `train.py` cho cả IDRiD và DRIVE; lưu checkpoint tốt nhất `best_model.pth`.

### Giai đoạn 2: Tối ưu AI & Tích hợp Full-Stack
- [ ] Xây dựng kiến trúc mô hình **Attention U-Net** (`attention_unet.py`) với Attention Gates để cải thiện phân vùng các tổn thương nhỏ (Microaneurysms - MA).
- [ ] Huấn luyện, đánh giá so sánh U-Net và Attention U-Net trên cùng tập validation/test độc lập.
- [ ] Xây dựng **Medical Evaluator** (`evaluator.py`): Đếm ổ tổn thương bằng Connected Components, tính diện tích pixel và tỷ lệ % trên diện tích võng mạc FOV.
- [ ] Xây dựng Frontend ReactJS (TypeScript + Tailwind CSS + Lucide Icons + Recharts).
- [ ] Xây dựng **Medical Image Canvas Viewer**:
  - 3 chế độ xem: Original, Ground Truth (nếu có), AI Prediction.
  - Điều khiển hiển thị: Bật/tắt từng lớp nhãn (MA: Đỏ, HE: Cam, EX: Vàng, SE: Xanh lá, OD: Xanh dương, Mạch máu: Tím).
  - Tương tác: Pan, Zoom, Reset, Opacity slider, So sánh Slider trước/sau.
- [ ] Tích hợp Celery Worker + Redis Queue cho xử lý phân tích AI bất đồng bộ (trạng thái: PENDING -> PROCESSING -> COMPLETED/FAILED).
- [ ] Tích hợp MinIO Object Storage (kèm local fallback storage bảo đảm hệ thống luôn chạy được offline).
- [ ] Xây dựng module kết xuất báo cáo y khoa dạng PDF chuyên nghiệp bằng **ReportLab**.

### Giai đoạn 3: Đóng gói Docker & Kiểm thử Toàn diện
- [ ] Tạo `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml` đa container (frontend, backend, celery worker, redis, postgres, minio).
- [ ] Cấu hình chạy linh hoạt trên cả **CPU** và **GPU (CUDA)**.
- [ ] Viết bộ kiểm thử tự động bằng `pytest` (AI Pipeline tests, Model forward tests, Metric calculation tests, Backend API tests, Evaluator tests).
- [ ] Tạo tài liệu kiểm thử chi tiết `TEST_CASES.md` với ID, Input, Expected Result, Actual Result, Pass/Fail.
- [ ] Soạn thảo hướng dẫn cài đặt và vận hành toàn diện trong `README.md`.

### Giai đoạn 4: Chuẩn bị Hồ sơ Bảo vệ Đồ án Tốt nghiệp / Đồ án Chuyên ngành
- [ ] Viết tài liệu báo cáo kỹ thuật 5 chương chuẩn format VKU trong thư mục `docs/`:
  - Chương 1: Tổng quan đề tài và tính cấp thiết trong chẩn đoán võng mạc tiểu đường.
  - Chương 2: Cơ sở lý thuyết về Deep Learning, U-Net, Attention Mechanism, Dice Loss, Connected Components.
  - Chương 3: Phân tích và thiết kế hệ thống (Kiến trúc, Use Cases, ERD, Sequence Diagram, API Specs).
  - Chương 4: Xây dựng hệ thống và kết quả thực nghiệm (Dữ liệu IDRiD, DRIVE, Kết quả so sánh U-Net vs Attention U-Net, Giao diện Web, Báo cáo PDF).
  - Chương 5: Đánh giá, kết luận, giới hạn và hướng phát triển.
- [ ] Chuẩn bị Slide thuyết trình 12–15 slide chuẩn tiếng Anh/tiếng Việt chuyên nghiệp.
- [ ] Kịch bản thuyết trình bảo vệ 10–15 phút và bộ câu hỏi phản biện chuyên sâu.

---

## 4. BẢNG TIÊU CHUẨN KỸ THUẬT VÀ QUY TẮC NGHIÊM NGẶT

| Thành phần | Tiêu chuẩn & Quy tắc thực hiện |
| :--- | :--- |
| **Tính chân thực của dữ liệu** | Không tạo số liệu giả, không tự bịa đặt Ground Truth. Dữ liệu huấn luyện xuất phát 100% từ IDRiD và DRIVE thực tế. |
| **Bảo toàn ranh giới dữ liệu** | Phân chia tập train/val/test rành mạch, không để ảnh gốc hoặc augmented versions bị rò rỉ (data leakage). |
| **Tách biệt nhiệm vụ** | IDRiD giải quyết tổn thương và đĩa thị (5 nhãn). DRIVE giải quyết cây mạch máu trong FOV mask. Không trộn lẫn hai bài toán. |
| **Độ tin cậy định lượng** | Chỉ số diện tích hiển thị rõ đơn vị pixel và phần trăm trên FOV. Chỉ quy đổi ra mm² khi có hiệu chuẩn quang học thiết bị chụp đáy mắt. |
| **Khả năng chạy độc lập** | Hệ thống hỗ trợ chế độ CPU lẫn GPU RTX 4050, có thể khởi chạy bằng cả Docker Compose lẫn chạy trực tiếp bằng Python/Vite. |

