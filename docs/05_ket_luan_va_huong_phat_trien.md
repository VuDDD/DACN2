# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  

---

## 5.1. KẾT QUẢ ĐẠT ĐƯỢC CỦA ĐỒ ÁN

Trải qua quá trình nghiên cứu lý thuyết, phân tích thiết kế và hiện thực hóa thực nghiệm, đồ án đã hoàn thành xuất sắc toàn diện các mục tiêu đặt ra:

### 5.1.1. Về mặt Trí tuệ Nhân tạo & Xử lý Ảnh Y sinh (AI & Computer Vision)
1. **Thiết kế kiến trúc hai nhánh chuyên biệt (Dual-Branch AI):** Phân tách độc lập quy trình phân vùng đa tổn thương (IDRiD: Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates, Optic Disc) và phân vùng cây mao mạch võng mạc (DRIVE), bảo đảm không trộn lẫn các đặc trưng có bản chất sinh học khác nhau.
2. **Xây dựng thành công Baseline U-Net và Attention U-Net:** Hiện thực hóa cơ chế cổng chú ý (Attention Gates) giúp mô hình tập trung tối đa vào các vùng tổn thương vi mạch có kích thước cực nhỏ, giảm thiểu nhiễu nền đáy mắt.
3. **Pipeline dữ liệu và tiền xử lý chuẩn y khoa:** Áp dụng kỹ thuật cân bằng độ tương phản thích ứng cục bộ (CLAHE) trên kênh màu Green, kỹ thuật tăng cường dữ liệu hình học đồng bộ giữa ảnh và mặt nạ (Synchronized Augmentations), và tự động trích xuất vùng thị trường võng mạc hợp lệ (FOV Mask).
4. **Hàm mất mát và đánh giá y sinh toàn diện:** Thiết kế hàm mất mát kết hợp `BCEDiceLoss` khắc phục triệt để bài toán mất cân bằng dữ liệu nghiêm trọng; bộ giải thuật `evaluator.py` ứng dụng Connected Components đếm chính xác số ổ tổn thương và định lượng diện tích theo pixel và tỷ lệ \% FOV.
5. **Huấn luyện và kiểm chứng thực nghiệm trên GPU RTX 4050:** Mô hình đã được huấn luyện trực tiếp trên tập dữ liệu thực tế, ghi nhận đầy đủ lịch sử suy giảm hàm mất mát và lưu trữ các điểm kiểm tra trọng số tốt nhất (`best_model.pth`).

### 5.1.2. Về mặt Kỹ thuật Phần mềm & Nền tảng Full-stack (Full-stack Platform)
1. **Backend RESTful API hoàn chỉnh:** Xây dựng bằng FastAPI với đầy đủ các phân hệ Quản lý Tài khoản (JWT Authentication), Quản lý Bệnh nhân (CRUD), Xử lý Ảnh y tế, Điều phối Tác vụ Phân tích, Thống kê Lâm sàng và Kết xuất Báo cáo.
2. **Hệ thống Xử lý Bất đồng bộ mạnh mẽ:** Tích hợp Celery Worker và Redis Message Broker, giải phóng Web Server khỏi các tác vụ suy luận nặng, hỗ trợ theo dõi tiến trình (Polling Status) mượt mà.
3. **Lưu trữ Đối tượng Chuẩn S3 & Độc lập Ngoại tuyến:** Tích hợp MinIO Object Storage đi kèm lớp trừu tượng lưu trữ đĩa cục bộ thông minh, bảo đảm hệ thống luôn sẵn sàng vận hành ổn định trong mọi điều kiện mạng.
4. **Giao diện Người dùng Chuyên nghiệp:** Phát triển bằng ReactJS, Vite, TypeScript và Tailwind CSS, nổi bật với **HTML5 Canvas Medical Viewer** hỗ trợ thu phóng, xoay chuyển, trượt chia màn hình trước/sau và bật/tắt từng lớp tổn thương độc lập.
5. **Kết xuất Báo cáo Y khoa PDF:** Sử dụng ReportLab tự động biên dịch phiếu kết quả xét nghiệm y khoa khổ A4 đạt chuẩn bệnh viện.
6. **Kiểm thử Toàn diện và Đóng gói Docker:** 14 ca kiểm thử tự động với Pytest đạt tỷ lệ thành công 100%; toàn bộ hệ thống được đóng gói thành các container Docker Compose sẵn sàng triển khai bằng một câu lệnh duy nhất.

---

## 5.2. CÁC HẠN CHẾ CỦA HỆ THỐNG HIỆN TẠI

Bên cạnh những kết quả tích cực đã đạt được, hệ thống vẫn tồn tại một số giới hạn cần được ghi nhận minh bạch:
1. **Kích thước mẫu dữ liệu huấn luyện:** Các bộ dữ liệu công khai có nhãn phân vùng mức pixel chuẩn chuyên gia (Pixel-level Ground Truth) như IDRiD (81 ảnh) và DRIVE (40 ảnh) có quy mô tương đối khiêm tốn. Mô hình cần được tiếp cận nguồn dữ liệu lâm sàng đa dạng hơn từ nhiều chủng tộc và nhiều dòng máy chụp đáy mắt khác nhau.
2. **Độ nhạy đối với chất lượng ảnh chụp:** Đối với các bức ảnh đáy mắt chụp trong điều kiện đồng tử giãn không đủ, có vệt mờ do đục thủy tinh thể nặng (Cataract) hoặc xuất hiện bóng phản chiếu quang học của thấu kính, ranh giới dự đoán tổn thương vi phình mạch có thể xuất hiện dương tính giả (False Positives).
3. **Giới hạn quy đổi vật lý ($\text{mm}^2$):** Do các bộ dữ liệu công khai không cung cấp thông số tiêu cự, khẩu độ và độ phóng đại quang học của camera đáy mắt, hệ thống chỉ báo cáo diện tích tổn thương dưới dạng pixel và tỷ lệ phần trăm trên FOV. Việc quy đổi sang milimet vuông tuyệt đối đòi hỏi dữ liệu hiệu chuẩn vật lý chuẩn xác của từng thiết bị chụp.

---

## 5.3. HƯỚNG PHÁT TRIỂN TIẾP THEO (FUTURE WORK)

Nhằm nâng cao độ chính xác và khả năng ứng dụng thực tế trong môi trường lâm sàng, các hướng nghiên cứu tiếp theo bao gồm:

1. **Nghiên cứu Kiến trúc Vision Transformer (ViT) & Mamba y sinh:**
   - Thử nghiệm các mô hình tiên tiến như TransUNet, Swin UNETR hoặc Vision Mamba (Vim) nhằm khai thác khả năng học tương quan ngữ cảnh toàn cục (Long-range spatial dependencies) mà không làm mất chi tiết cục bộ.
2. **Áp dụng Kỹ thuật Học Tự Giám Sát (Self-Supervised Learning - SSL):**
   - Tận dụng hàng trăm nghìn bức ảnh đáy mắt không nhãn từ các cơ sở dữ liệu lớn (EyePACS, Kaggle DR) để tiền huấn luyện (Pre-training) bằng phương pháp Masked Autoencoders (MAE) hoặc DINOv2 trước khi tinh chỉnh (Fine-tuning) trên IDRiD.
3. **Phát triển Module Tự Động Phân Độ Bệnh Học (Automatic Grading):**
   - Kết hợp kết quả định lượng diện tích tổn thương với thang đo lâm sàng quốc tế ICDR (International Clinical Diabetic Retinopathy Disease Severity Scale) để đưa ra khuyến nghị phân độ từ nhẹ (Mild), trung bình (Moderate), nặng (Severe NPDR) đến tăng sinh (PDR).
4. **Tối ưu hóa Mô hình cho Thiết bị Di động & Biên (Edge AI Deployment):**
   - Áp dụng các kỹ thuật lượng tử hóa (Quantization INT8), cắt tỉa mạng (Pruning) và chưng cất tri thức (Knowledge Distillation) để nhúng mô hình vào các thiết bị chụp đáy mắt cầm tay gắn liền với smartphone, hỗ trợ các đội ngũ y tế lưu động tại vùng sâu vùng xa.
5. **Hợp tác Thử nghiệm Lâm sàng:**
   - Phối hợp với các bệnh viện mắt và trung tâm nội tiết đái tháo đường để tiến hành kiểm nghiệm mù đôi (Blind Trial), so sánh độ chính xác giữa kết quả hệ thống AI và đánh giá của hội đồng bác sĩ nhãn khoa chuyên khoa II.
