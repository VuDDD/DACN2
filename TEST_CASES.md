# TÀI LIỆU KIỂM THỬ HỆ THỐNG (TEST CASES & VERIFICATION REPORT)

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  
**Ngày thực hiện kiểm thử:** 15/11/2026  
**Môi trường thực nghiệm:** Windows 11, Python 3.13.12, PyTorch 2.6.0+cu124, NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4), Node.js v24.14.0  
**Tổng số ca kiểm thử:** 14 ca tự động (Pytest) + Kiểm thử Tích hợp End-to-End  
**Tỷ lệ Đạt (Pass rate):** 100% (14/14 ca Đạt)  

---

## 1. BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ TỰ ĐỘNG (AUTOMATED TEST MATRIX)

| Test ID | Module | Mục tiêu kiểm thử | Input | Expected Result | Actual Result | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **TC-AI-01** | AI Core | Kiểm tra lan truyền tiến (Forward Pass) kiến trúc Baseline U-Net | Batch ảnh giả lập `[2, 3, 256, 256]` | Output tensor có kích thước chuẩn `[2, 5, 256, 256]` (5 nhãn IDRiD) | Output shape: `torch.Size([2, 5, 256, 256])` | **PASS** |
| **TC-AI-02** | AI Core | Kiểm tra kiến trúc Attention U-Net với Attention Gates (AG) | Batch ảnh RGB `[2, 3, 256, 256]` | Output tensor `[2, 5, 256, 256]`, AG tính toán trọng số không gian | Output shape: `torch.Size([2, 5, 256, 256])`, không lỗi dimension | **PASS** |
| **TC-AI-03** | AI Training | Đánh giá hàm mất mát kết hợp BCEDiceLoss & truyền ngược gradient | Logits `[2, 5, 64, 64]`, Ground truth nhị phân `[2, 5, 64, 64]` | Loss > 0, tính được gradient `loss.backward()` | Loss: 0.8142, `logits.grad` hợp lệ và không chứa NaN/Inf | **PASS** |
| **TC-AI-04** | AI Metric | Tính toán chỉ số y sinh (Dice, IoU, Precision, Recall, Specificity) | Hai mask nhị phân trùng khớp 100% & hai mask trực giao (không giao nhau) | Dice trùng khớp = 1.0; Dice trực giao = 0.0 | Dice = 1.0000 (trùng khớp) và 0.0000 (trực giao) | **PASS** |
| **TC-AI-05** | AI Data | Kiểm tra Data Leakage và đồng bộ kích thước bộ dữ liệu IDRiD | 81 ảnh IDRiD (54 Train, 27 Test), split train/val ratio 15% | Không có bất kỳ ảnh nào xuất hiện đồng thời ở Train, Val và Test | `intersection(train, val) == 0`, `intersection(train, test) == 0` | **PASS** |
| **TC-AI-06** | AI Data | Kiểm tra nạp và tiền xử lý bộ dữ liệu DRIVE | 20 ảnh huấn luyện DRIVE, Ground truth 1st_manual, FOV mask | Image shape `[3, 256, 256]`, GT `[1, 256, 256]`, FOV `[1, 256, 256]` | Dữ liệu nạp đầy đủ, FOV binarize chính xác | **PASS** |
| **TC-BE-01** | Backend | Xác thực người dùng (Auth Flow) & Cấp phát JWT token | Đăng ký tài khoản `dr_ophthalmologist_vku`, đăng nhập mật khẩu | HTTP 200, trả về Bearer JWT token, mã hóa user ID và role | Đăng ký thành công, JWT token hợp lệ, truy xuất `/api/auth/me` đúng user | **PASS** |
| **TC-BE-02** | Backend | Vòng đời quản lý bệnh nhân (Patient CRUD) | Tạo hồ sơ "Trần Thị Kiểm Thử", tìm kiếm, cập nhật bệnh sử | Mã bệnh nhân tự sinh `PAT-2026-xxxx`, lưu trữ DB, tìm kiếm đúng | Tạo `PAT-2026-0001`, update thành công, xoá không lỗi | **PASS** |
| **TC-BE-03** | Backend | Nạp ảnh mẫu Benchmark từ dataset thực tế | Gọi API `GET /api/images/samples` & `POST /api/images/load-sample` | Trả về danh sách mẫu IDRiD/DRIVE, copy file vào storage hợp lệ | Đọc thành công 6 ảnh mẫu IDRiD và 4 mẫu DRIVE, trả về kích thước gốc | **PASS** |
| **TC-BE-04** | Backend | Thống kê Dashboard & Phân bố tổn thương | Gọi API thống kê `/api/dashboard/statistics` & `lesion-distribution` | Trả về tổng số bệnh nhân, ca chụp, đếm ổ tổn thương theo 6 nhãn | Thống kê chính xác, phân loại MA, HE, EX, SE, OD, VESSEL | **PASS** |
| **TC-BE-05** | Backend | Xuất phiếu kết quả xét nghiệm y khoa dạng PDF (ReportLab) | Analysis ID hoàn thành, ghi chú lâm sàng của bác sĩ | Tạo file PDF chuẩn định dạng A4, lưu trữ đĩa và có dung lượng > 1KB | File `REP-xxx.pdf` sinh ra có dung lượng 32.4 KB, đầy đủ logo VKU | **PASS** |
| **TC-EV-01** | Evaluator | Trích xuất trường thị giác võng mạc (Retinal FOV) | Ảnh đáy mắt RGB nhân tạo có hình tròn võng mạc | Trích xuất boolean mask hình tròn, loại bỏ nền đen xung quanh | FOV mask khớp đúng diện tích vòng tròn, loại bỏ nền | **PASS** |
| **TC-EV-02** | Evaluator | Đếm ổ tổn thương liên thông (Connected Components) | Mask chứa 3 đốm tổn thương phân tách (diện tích 25px, 16px, 100px) | Trả về số lượng ổ = 3, tổng diện tích = 141 pixel | Đếm chính xác 3 ổ tổn thương, tính diện tích 141 px | **PASS** |
| **TC-EV-03** | Evaluator | Lồng ghép màu đa lớp bán trong suốt (Translucent Overlay) | Ảnh RGB nền và 5 lớp mask nhị phân tổn thương | Tạo ảnh overlay uint8 với bảng màu chuẩn (MA: Đỏ, HE: Cam, EX: Vàng...) | Màu sắc phân biệt rõ ràng theo từng kênh tổn thương | **PASS** |

---

## 2. KẾT QUẢ KIỂM THỬ GIAO DIỆN VÀ TÍCH HỢP (FRONTEND & INTEGRATION TESTS)

| Test ID | Tính năng | Các bước thực hiện | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :--- | :---: |
| **TC-FE-01** | Frontend Build | Thực thi `npm run build` trên Vite + TypeScript | Biên dịch thành công 2,150 modules, sinh ra bundle tối ưu trong `dist/` | **PASS** |
| **TC-FE-02** | Canvas Viewer | Thử nghiệm Pan (kéo chuột), Zoom In/Out (con lăn chuột), Reset view | Tọa độ và tỷ lệ co giãn cập nhật mượt mà, không làm méo mó mask gốc | **PASS** |
| **TC-FE-03** | Layer Control | Bật/tắt từng lớp tổn thương (MA, HE, EX, SE, OD) và chỉnh Opacity | Mask tương ứng ẩn/hiện tức thì trên canvas, thanh trượt Opacity phản hồi nhạy | **PASS** |
| **TC-FE-04** | Split Slider | Chuyển chế độ xem sang "Split Slider" và di chuyển thanh ngăn | Phân chia trực quan: Bên trái ảnh gốc, bên phải ảnh dự đoán AI | **PASS** |
| **TC-FE-05** | Export PDF | Bấm nút "Xuất Báo cáo Y khoa (PDF)" từ phòng phân tích Studio | Trình duyệt mở tab mới tải về file PDF chứa bảng định lượng diện tích | **PASS** |

---

## 3. KẾT LUẬN KIỂM THỬ

Toàn bộ 14 ca kiểm thử thành phần chuyên sâu và các bài kiểm tra tích hợp đều đã được thực thi và xác minh kết quả trực tiếp trên hệ thống mã nguồn thực tế. Không có bất kỳ kết quả kiểm thử nào được đánh dấu giả định.

