# CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  

---

## 3.1. KIẾN TRÚC TỔNG THỂ HỆ THỐNG (SYSTEM ARCHITECTURE)

Hệ thống được thiết kế theo mô hình Microservices phân tách rành mạch giữa giao diện người dùng, dịch vụ API, hàng đợi xử lý tác vụ nặng và tầng suy luận Deep Learning.

```mermaid
flowchart TD
    subgraph Client["Lớp Giao Diện (Frontend SPA)"]
        UI["ReactJS + TypeScript + Tailwind CSS"]
        Canvas["HTML5 Canvas Medical Viewer"]
        Charts["Recharts Dashboard"]
    end

    subgraph Gateway["Lớp Dịch Vụ Cốt Lõi (Backend API)"]
        API["FastAPI REST Server (Port 8000)"]
        Auth["JWT Security & RBAC Guard"]
        StorageSvc["Storage Service Adapter"]
    end

    subgraph DataStore["Lớp Lưu Trữ Dữ Liệu"]
        PG[("PostgreSQL Database (Port 5432)")]
        Redis[("Redis Message Broker (Port 6379)")]
        MinIO[("MinIO S3 Object Storage (Port 9000)")]
    end

    subgraph Worker["Lớp Xử Lý Bất Đồng Bộ & AI Core"]
        Celery["Celery Task Worker"]
        AI["AI Inference Engine (PyTorch)"]
        Evaluator["Medical Evaluator (Connected Components)"]
        PDF["ReportLab PDF Engine"]
    end

    UI -->|HTTP / JSON + JWT| API
    Canvas -->|Render Binary Buffers| UI
    API -->|Authenticate| Auth
    API -->|Read / Write Metadata| PG
    API -->|Push Task Payload| Redis
    API -->|Upload / Serve Images| StorageSvc
    StorageSvc -->|S3 API / Disk Fallback| MinIO

    Redis -->|Consume Task| Celery
    Celery -->|Run Prediction| AI
    AI -->|Output Masks| Evaluator
    Evaluator -->|Calculate Area & Foci| Celery
    Celery -->|Update Status & Metrics| PG
    Celery -->|Save Mask Overlays| StorageSvc
    Celery -->|Generate Report| PDF
    PDF -->|Save PDF File| StorageSvc
```

---

## 3.2. BIỂU ĐỒ VÀ ĐẶC TẢ USE CASE (USE CASE SPECIFICATION)

### 3.2.1. Tác nhân hệ thống (Actors)
1. **Bác sĩ Nhãn khoa / Nhà nghiên cứu (Doctor / Researcher):**
   - Đăng nhập, đăng ký tài khoản.
   - Quản lý hồ sơ bệnh nhân (Thêm, sửa, tìm kiếm bệnh án).
   - Tải lên ảnh đáy mắt võng mạc hoặc chọn mẫu dữ liệu chuẩn IDRiD/DRIVE.
   - Lựa chọn mô hình Deep Learning (U-Net hoặc Attention U-Net) và loại nhiệm vụ (Tổn thương hoặc Mạch máu).
   - Kích hoạt tiến trình phân tích bất đồng bộ và theo dõi tiến độ.
   - Thao tác trên HTML5 Canvas Viewer (Phóng to, thu nhỏ, di chuyển, trượt chia màn hình Split Slider, bật tắt nhãn tổn thương, chỉnh Opacity).
   - Xem bảng chỉ số định lượng diện tích và đếm số ổ tổn thương.
   - Kết xuất và tải phiếu kết quả xét nghiệm định dạng PDF.
2. **Quản trị viên Hệ thống (System Administrator):**
   - Toàn quyền quản trị tài khoản người dùng, phân quyền truy cập.
   - Giám sát tình trạng hạ tầng (Database, Redis Queue, MinIO, GPU CUDA).
   - Đánh giá tổng hợp chỉ số hiệu năng thực nghiệm của các phiên bản mô hình AI.

```mermaid
flowchart LR
    Doctor((Bác sĩ / Doctor))
    Admin((Quản trị / Admin))

    subgraph System["Nền tảng Retina AI"]
        UC1["Đăng nhập / Đăng ký (JWT)"]
        UC2["Quản lý Bệnh nhân (CRUD)"]
        UC3["Tải ảnh / Chọn mẫu Benchmark"]
        UC4["Cấu hình & Chạy Phân tích AI"]
        UC5["Quan sát Tương tác Đa lớp (Canvas)"]
        UC6["Định lượng Diện tích & Đếm Ổ"]
        UC7["Xuất Báo cáo Y khoa PDF"]
        UC8["Theo dõi Thống kê Dashboard"]
        UC9["Quản trị Người dùng & Phân quyền"]
    end

    Doctor --> UC1
    Doctor --> UC2
    Doctor --> UC3
    Doctor --> UC4
    Doctor --> UC5
    Doctor --> UC6
    Doctor --> UC7
    Doctor --> UC8

    Admin --> UC1
    Admin --> UC8
    Admin --> UC9
```

---

## 3.3. THIẾT KẾ CƠ SỞ DỮ LIỆU (DATABASE ERD & SCHEMAS)

Cơ sở dữ liệu được thiết kế theo chuẩn 3NF (Third Normal Form), bảo đảm tính toàn vẹn tham chiếu, tối ưu hóa tốc độ truy vấn qua các chỉ mục (Index) và quan hệ chặt chẽ giữa các thực thể.

```mermaid
erDiagram
    USERS ||--o{ ANALYSES : performs
    PATIENTS ||--o{ ANALYSES : owns
    ANALYSES ||--o{ LESION_DETAILS : contains
    ANALYSES ||--o{ REPORTS : generates

    USERS {
        int id PK
        string username UK
        string email UK
        string password_hash
        string full_name
        string role
        boolean is_active
        datetime created_at
        datetime updated_at
    }

    PATIENTS {
        int id PK
        string patient_code UK
        string full_name
        string gender
        date date_of_birth
        string phone_number
        text medical_history
        text notes
        datetime created_at
        datetime updated_at
    }

    ANALYSES {
        int id PK
        int patient_id FK
        int user_id FK
        string task_type
        string model_name
        string analysis_status
        text error_message
        string original_image_url
        string predicted_mask_url
        string ground_truth_mask_url
        string fov_mask_url
        int retinal_fov_pixels
        int lesion_area_pixels
        float lesion_percentage
        int processing_time_ms
        float dice_score
        float iou_score
        float precision
        float recall
        float specificity
        float f1_score
        datetime created_at
        datetime updated_at
    }

    LESION_DETAILS {
        int id PK
        int analysis_id FK
        string lesion_type
        int count
        int area_pixels
        float area_percentage
    }

    REPORTS {
        int id PK
        int analysis_id FK
        string report_code UK
        string pdf_path
        text doctor_notes
        datetime generated_at
    }
```

### Chi tiết các thực thể:
1. **`users`:** Lưu thông tin tài khoản bác sĩ và quản trị viên, mật khẩu được băm an toàn bằng thuật toán bcrypt.
2. **`patients`:** Lưu trữ lý lịch y bạ của bệnh nhân (mã định danh tự động `PAT-YYYY-XXXX`, họ tên, ngày sinh, giới tính, tiền sử bệnh lý tiểu đường/tim mạch).
3. **`analyses`:** Thực thể trung tâm lưu trữ metadata ca phân tích ảnh đáy mắt, trạng thái xử lý (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`), đường dẫn ảnh và các chỉ số đo lường lâm sàng.
4. **`lesion_details`:** Bảng chi tiết hóa định lượng cho từng loại tổn thương (MA, HE, EX, SE, OD, VESSEL): số ổ tổn thương liên thông (count), tổng diện tích pixel và tỷ lệ phần trăm trên trường thị giác FOV.
5. **`reports`:** Quản lý các phiếu kết quả xét nghiệm dạng file PDF đã kết xuất.

---

## 3.4. BIỂU ĐỒ TUẦN TỰ (SEQUENCE DIAGRAMS)

### 3.4.1. Quy trình Tải ảnh và Thực thi Phân tích AI Bất đồng bộ

```mermaid
sequenceDiagram
    autonumber
    actor Doctor as Bác sĩ (Doctor)
    participant Client as ReactJS Frontend
    participant API as FastAPI Backend
    participant Storage as Storage / MinIO
    participant Queue as Redis Queue
    participant Worker as Celery AI Worker
    participant DB as PostgreSQL

    Doctor->>Client: 1. Chọn bệnh nhân & tải ảnh võng mạc
    Client->>API: 2. POST /api/images/upload (FormData)
    API->>Storage: 3. Lưu ảnh gốc & kiểm tra kích thước
    Storage-->>API: 4. Trả về đường dẫn lưu trữ
    API-->>Client: 5. Trả về image_url và resolution

    Doctor->>Client: 6. Bấm "Bắt đầu Phân tích AI"
    Client->>API: 7. POST /api/analysis/predict (patient_id, image_url, model, task)
    API->>DB: 8. Tạo bản ghi Analysis (Status: PENDING)
    API->>Queue: 9. Đẩy tác vụ vào hàng đợi Celery
    API-->>Client: 10. HTTP 202 Accepted (analysis_id)

    loop Polling tiến độ phân tích
        Client->>API: 11. GET /api/analysis/{id}/status
        API->>DB: 12. Đọc trạng thái hiện tại
        API-->>Client: 13. Trả về Status (PENDING/PROCESSING/COMPLETED)
    end

    Queue->>Worker: 14. Phân phối tác vụ cho AI Worker
    Worker->>DB: 15. Cập nhật Status: PROCESSING
    Worker->>Storage: 16. Đọc ảnh gốc từ bộ nhớ
    Worker->>Worker: 17. Chạy mô hình U-Net / Attention U-Net
    Worker->>Worker: 18. Medical Evaluator (Connected Components & FOV %)
    Worker->>Storage: 19. Lưu ảnh mặt nạ phân vùng (Predicted Overlay)
    Worker->>DB: 20. Lưu kết quả định lượng & Status: COMPLETED

    Client->>API: 21. GET /api/analysis/{id}/result
    API->>DB: 22. Đọc toàn bộ kết quả phân tích & tổn thương chi tiết
    API-->>Client: 23. Trả về kết quả hoàn chỉnh
    Client->>Doctor: 24. Hiển thị tương tác trên Canvas Viewer & bảng số liệu
```

### 3.4.2. Quy trình Kết xuất và Tải Phiếu Kết quả PDF

```mermaid
sequenceDiagram
    autonumber
    actor Doctor as Bác sĩ
    participant Client as ReactJS Frontend
    participant API as FastAPI Backend
    participant Generator as ReportLab PDF Engine
    participant Storage as Storage / MinIO
    participant DB as PostgreSQL

    Doctor->>Client: 1. Bấm "Xuất Báo cáo Y khoa (PDF)"
    Client->>API: 2. POST /api/reports/generate/{analysis_id} (doctor_notes)
    API->>DB: 3. Lấy thông tin ca phân tích, bệnh nhân, bác sĩ
    API->>Storage: 4. Lấy file ảnh gốc và ảnh mask dự đoán
    API->>Generator: 5. Biên dịch mẫu PDF (Tiêu đề, Bảng định lượng, Ảnh, Cảnh báo y tế)
    Generator->>Storage: 6. Ghi file PDF (REP-XXXXXX.pdf)
    API->>DB: 7. Lưu bản ghi Report vào database
    API-->>Client: 8. Trả về thông tin Report & URL tải về
    Client->>API: 9. GET /api/reports/{id}/download
    API-->>Doctor: 10. Tải trực tiếp file PDF chính thức về máy
```

---

## 3.5. THIẾT KẾ RESTFUL API (API SPECIFICATION)

Hệ thống cung cấp giao diện lập trình ứng dụng RESTful API chuẩn mực theo chuẩn OpenAPI v3:

| Phương thức | Đường dẫn (Endpoint) | Quyền hạn | Mô tả chức năng |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Public | Đăng ký tài khoản bác sĩ / nghiên cứu viên |
| `POST` | `/api/auth/login-json` | Public | Đăng nhập hệ thống, nhận mã JWT Bearer Token |
| `GET` | `/api/auth/me` | Doctor/Admin | Lấy thông tin hồ sơ tài khoản hiện tại |
| `GET` | `/api/patients` | Doctor/Admin | Lấy danh sách bệnh nhân có phân trang và tìm kiếm |
| `POST` | `/api/patients` | Doctor/Admin | Tạo mới hồ sơ bệnh nhân (mã PAT tự động) |
| `GET` | `/api/patients/{id}` | Doctor/Admin | Lấy chi tiết bệnh nhân và các ca chụp liên quan |
| `PUT` | `/api/patients/{id}` | Doctor/Admin | Chỉnh sửa thông tin hành chính, bệnh sử |
| `DELETE` | `/api/patients/{id}` | Doctor/Admin | Xóa hồ sơ bệnh nhân và toàn bộ lịch sử phân tích |
| `POST` | `/api/images/upload` | Doctor/Admin | Tải ảnh võng mạc lên máy chủ, kiểm định MIME type |
| `GET` | `/api/images/samples` | Public | Danh sách ảnh mẫu thực tế từ IDRiD và DRIVE |
| `POST` | `/api/images/load-sample` | Doctor/Admin | Nạp ảnh mẫu benchmark vào không gian phân tích |
| `GET` | `/api/images/serve/{sub}/{fn}` | Public | Phục vụ file ảnh, mặt nạ, tài liệu PDF trực tiếp |
| `POST` | `/api/analysis/predict` | Doctor/Admin | Kích hoạt tác vụ phân tích và định lượng AI |
| `GET` | `/api/analysis/{id}` | Doctor/Admin | Lấy toàn bộ kết quả phân tích và chi tiết tổn thương |
| `GET` | `/api/analysis/{id}/status` | Doctor/Admin | Điểm kiểm tra trạng thái tác vụ phục vụ Polling |
| `POST` | `/api/reports/generate/{id}` | Doctor/Admin | Sinh file PDF phiếu kết quả xét nghiệm y khoa |
| `GET` | `/api/reports/{id}/download` | Doctor/Admin | Tải file PDF báo cáo đính kèm |
| `GET` | `/api/dashboard/statistics` | Doctor/Admin | Thống kê số lượng bệnh nhân, ca chụp, tỷ lệ hoàn thành |
| `GET` | `/api/dashboard/lesion-distribution` | Doctor/Admin | Thống kê phân bố số lượng và diện tích theo 6 nhãn |
| `GET` | `/api/dashboard/model-performance` | Doctor/Admin | Đánh giá hiệu năng thực nghiệm U-Net vs Attention U-Net |
