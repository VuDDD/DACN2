# CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  

---

## 1.1. ĐẶT VẤN ĐỀ

Bệnh võng mạc tiểu đường (Diabetic Retinopathy - DR) là một trong những nguyên nhân hàng đầu gây suy giảm thị lực nghiêm trọng và mù lòa không thể đảo ngược ở nhóm dân số trong độ tuổi lao động trên toàn cầu. Theo thống kê của Tổ chức Y tế Thế giới (WHO) và Liên đoàn Đái tháo đường Quốc tế (IDF), ước tính có hơn 537 triệu người trưởng thành đang sống chung với bệnh đái tháo đường, và khoảng một phần ba trong số họ sẽ xuất hiện các biểu hiện tổn thương vi mạch tại đáy mắt ở các mức độ khác nhau.

Trong thực hành lâm sàng nhãn khoa, việc phát hiện sớm và can thiệp kịp thời (bằng kiểm soát đường huyết, laser quang đông hoặc tiêm nội nhãn kháng VEGF) có thể ngăn ngừa tới 95% nguy cơ mất thị lực nghiêm trọng. Phương pháp sàng lọc tiêu chuẩn hiện nay dựa trên việc chụp ảnh màu đáy mắt (Color Fundus Photography - CFP). Tuy nhiên, việc đánh giá thủ công ảnh chụp đáy mắt đối mặt với nhiều thách thức to lớn:
1. **Sự quá tải của đội ngũ bác sĩ chuyên khoa mắt:** Tỷ lệ bác sĩ nhãn khoa trên quy mô dân số, đặc biệt tại các nước đang phát triển và vùng sâu vùng xa, là rất thấp. Số lượng người bệnh tiểu đường cần sàng lọc định kỳ hàng năm vượt xa năng lực thăm khám trực tiếp.
2. **Kích thước tổn thương vi mô cực nhỏ và tính đa dạng:** Các dấu hiệu tổn thương sớm nhất như vi phình mạch (Microaneurysms - MA) chỉ có đường kính từ vài pixel đến vài chục micromet, dễ bị bỏ sót do mỏi mắt hoặc điều kiện hiển thị không đồng nhất.
3. **Tính chủ quan giữa các người đọc ảnh (Inter-observer variability):** Mức độ phân loại và ranh giới tổn thương phụ thuộc lớn vào kinh nghiệm lâm sàng của từng bác sĩ, dẫn đến sự thiếu đồng nhất trong kết luận và khó theo dõi chính xác tiến triển điều trị qua thời gian.
4. **Thiếu công cụ định lượng khách quan:** Hầu hết các chẩn đoán hiện nay dừng lại ở mức định tính (ví dụ: "có xuất huyết rải rác", "xuất tiết cứng vùng hoàng điểm") mà không đo lường được diện tích tổn thương chính xác (pixel hoặc tỷ lệ phần trăm trên diện tích võng mạc hợp lệ), gây khó khăn cho việc đánh giá định lượng hiệu quả của phác đồ điều trị.

Do đó, việc nghiên cứu và xây dựng một **Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng Deep Learning** có khả năng xác định chính xác ranh giới tổn thương pixel-level, phân tách từng dạng tổn thương bệnh học và cung cấp các chỉ số định lượng khách quan là một nhu cầu cấp thiết, mang ý nghĩa khoa học và thực tiễn sâu sắc.

---

## 1.2. TÍNH CẤP THIẾT CỦA ĐỀ TÀI

1. **Ý nghĩa y tế cộng đồng:** Hỗ trợ quy trình sàng lọc diện rộng (mass screening), giúp phát hiện sớm bệnh võng mạc tiểu đường ở giai đoạn không tăng sinh (NPDR), giảm thiểu áp lực cho các bệnh viện tuyến trên và tiết kiệm chi phí điều trị cho xã hội.
2. **Định lượng thay vì chỉ phân loại (Segmentation & Quantification vs. Classification):** Các hệ thống AI thế hệ trước chủ yếu giải quyết bài toán phân loại nhị phân (Binary Classification: có bệnh / không bệnh) hoặc chấm điểm mức độ (Grading 0-4). Tuy nhiên, bác sĩ nhãn khoa cần biết chính xác *vị trí*, *hình thái*, *số lượng ổ* và *diện tích* của từng loại tổn thương riêng biệt (MA, HE, EX, SE) và cây mạch máu để quyết định phương án can thiệp.
3. **Tính toàn diện của sản phẩm phần mềm y tế:** Không dừng lại ở các đoạn script huấn luyện mô hình rời rạc, đồ án hướng đến một nền tảng hoàn chỉnh có kiến trúc Microservices hiện đại: xử lý ảnh đa lớp trên HTML5 Canvas, hàng đợi bất đồng bộ Celery + Redis, lưu trữ đối tượng MinIO, bảo mật kiểm soát truy cập vai trò (RBAC) và tự động xuất phiếu kết quả xét nghiệm dạng PDF.

---

## 1.3. MỤC TIÊU NGHIÊN CỨU

### 1.3.1. Mục tiêu khoa học & Trí tuệ nhân tạo (AI Core)
- Nghiên cứu và xây dựng kiến trúc mạng nơ-ron tích chập phân vùng ảnh y sinh: **Baseline U-Net** và kiến trúc cải tiến **Attention U-Net** tích hợp cơ chế cổng chú ý (Attention Gates).
- Xây dựng quy trình xử lý dữ liệu chuẩn hóa, tăng cường dữ liệu hình học đồng bộ (Synchronized Augmentations) và kỹ thuật cân bằng độ tương phản thích ứng cục bộ (CLAHE) trên kênh màu Green.
- Thiết kế hàm mất mát kết hợp `BCEDiceLoss` nhằm giải quyết bài toán mất cân bằng nghiêm trọng giữa diện tích tổn thương và nền đáy mắt.
- Huấn luyện, đánh giá và so sánh thực nghiệm độc lập trên hai bộ dữ liệu chuẩn quốc tế:
  - **IDRiD:** Phân vùng đa nhãn tổn thương võng mạc (MA, HE, EX, SE) và đĩa thị (OD).
  - **DRIVE:** Phân vùng hệ thống mao mạch võng mạc trong trường thị giác (FOV).

### 1.3.2. Mục tiêu kỹ thuật & Công nghệ phần mềm (Full-stack Platform)
- Thiết kế và lập trình Backend RESTful API hiệu năng cao bằng **Python FastAPI**, xác thực bằng **JWT Authentication**, phân quyền Admin và Doctor.
- Thiết kế cơ sở dữ liệu quan hệ **PostgreSQL** chuẩn hóa với các ràng buộc khóa ngoại, index tìm kiếm và di chuyển lược đồ tự động.
- Xây dựng hệ thống hàng đợi tác vụ bất đồng bộ **Celery + Redis**, phân tách luồng xử lý suy luận Deep Learning nặng khỏi luồng xử lý Web chính.
- Phát triển giao diện người dùng chuyên nghiệp bằng **ReactJS, Vite, TypeScript và Tailwind CSS**, nổi bật với **HTML5 Canvas Medical Viewer** hỗ trợ Pan, Zoom, trượt so sánh Split Slider và bật/tắt từng lớp nhãn màu độc lập.
- Xây dựng bộ công cụ phân tích y sinh **Medical Evaluator** ứng dụng giải thuật Connected Components để đếm số ổ tổn thương và tính tỷ lệ phần trăm diện tích so với trường thị giác FOV.
- Tự động xuất phiếu kết quả xét nghiệm y khoa dạng PDF bằng **ReportLab**.
- Đóng gói toàn bộ hệ thống bằng **Docker và Docker Compose**, bảo đảm tính di động và khả năng vận hành linh hoạt trên cả CPU và GPU CUDA.

---

## 1.4. ĐỐI TƯỢNG VÀ PHẠM VI NGHIÊN CỨU

### 1.4.1. Đối tượng nghiên cứu
- Ảnh màu đáy mắt kỹ thuật số (Color Fundus Photographs - CFP).
- Các dạng tổn thương vi mạch trong bệnh võng mạc đái tháo đường:
  - **Microaneurysms (MA):** Vi phình mạch, dạng chấm đỏ nhỏ hình tròn/bầu dục (10–100 μm).
  - **Haemorrhages (HE):** Xuất huyết võng mạc dạng đốm (blot) hoặc dạng ngọn lửa (flame-shaped).
  - **Hard Exudates (EX):** Xuất tiết cứng, các mảng lắng đọng lipid màu vàng có ranh giới rõ ràng.
  - **Soft Exudates / Cotton Wool Spots (SE):** Xuất tiết mềm, các đám mờ màu trắng xám do tắc nghẽn vi tuần hoàn lớp sợi thần kinh.
  - **Optic Disc (OD):** Đĩa thị giác (gai thị), điểm mốc giải phẫu tự nhiên quan trọng.
  - **Blood Vessels (Mạch máu võng mạc):** Mạng lưới động mạch, tĩnh mạch và mao mạch đáy mắt.
- Các kiến trúc Deep Learning phân vùng ảnh ngữ nghĩa (Semantic Segmentation) và cơ chế Attention Mechanism.

### 1.4.2. Phạm vi nghiên cứu
- **Dữ liệu thực nghiệm:** Giới hạn trên hai bộ dữ liệu tiêu chuẩn quốc tế đã được thẩm định chuyên gia: Indian Diabetic Retinopathy Image Dataset (**IDRiD**) và Digital Retinal Images for Vessel Extraction (**DRIVE**).
- **Phạm vi kỹ thuật:** Xây dựng hệ sinh thái phần mềm nguyên mẫu hoàn chỉnh (Working Prototype) có khả năng triển khai thực tế trên máy tính cá nhân hoặc máy chủ bệnh viện nội bộ.
- **Tuyên bố y đức và giới hạn:** Hệ thống được thiết kế như một công cụ hỗ trợ quyết định lâm sàng (Clinical Decision Support System - CDSS) và nghiên cứu khoa học; kết quả phân tích không thay thế chẩn đoán sau cùng của bác sĩ chuyên khoa mắt có chứng chỉ hành nghề.

