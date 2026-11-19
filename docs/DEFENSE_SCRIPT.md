# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN (10–15 PHÚT)

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  
**Thời lượng:** 12 – 15 phút (bao gồm Demo)  

---

## 1. MỞ ĐẦU (1.5 PHÚT)

> *"Kính thưa quý Thầy, Cô trong Hội đồng chấm Đồ án chuyên ngành 2, Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU).*
> 
> *Em xin phép được đại diện nhóm nghiên cứu trình bày báo cáo đồ án với đề tài: **'Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)'**.*
> 
> *Thưa quý Thầy Cô, bệnh võng mạc tiểu đường (Diabetic Retinopathy) hiện là nguyên nhân hàng đầu gây mù lòa không thể phục hồi ở nhóm tuổi lao động. Ước tính có hơn 537 triệu người mắc tiểu đường trên thế giới, và 1/3 trong số đó có biến chứng tại đáy mắt.*
> 
> *Tuy nhiên, hiện nay việc sàng lọc thủ công gặp 3 nút thắt rất lớn:*
> 1. *Sự thiếu hụt trầm trọng của đội ngũ bác sĩ nhãn khoa chuyên khoa sâu.*
> 2. *Các tổn thương giai đoạn sớm như vi phình mạch (Microaneurysms) có kích thước cực nhỏ, chỉ từ 10 đến 50 micromet, tức chỉ vài pixel trên ảnh, rất dễ bị bỏ sót.*
> 3. *Các hệ thống AI hiện nay hầu hết chỉ dừng lại ở mức **phân loại (Classification)** có bệnh hay không, mà **chưa định lượng được diện tích tổn thương và vị trí cụ thể**.*
> 
> *Xuất phát từ bài toán thực tế đó, mục tiêu của đề tài chúng em là xây dựng một **nền tảng Full-stack hoàn chỉnh**, kết hợp mô hình Deep Learning tiên tiến để phân vùng ở cấp độ điểm ảnh (pixel-level) và cung cấp các chỉ số định lượng khách quan hỗ trợ bác sĩ lâm sàng."*

---

## 2. DỮ LIỆU VÀ TRÍ TUỆ NHÂN TẠO (AI CORE) (3.5 PHÚT)

> *"Về mặt dữ liệu, đồ án tuân thủ nghiêm ngặt nguyên tắc **Tính chân thực của dữ liệu y khoa**, sử dụng 100% dữ liệu thực nghiệm đã qua thẩm định quốc tế:*
> - *Thứ nhất là bộ dữ liệu **IDRiD** gồm 81 ảnh đáy mắt độ phân giải cao, với nhãn Ground Truth 5 lớp tổn thương: Vi phình mạch (MA), Xuất huyết (HE), Xuất tiết cứng (EX), Xuất tiết mềm (SE) và Đĩa thị giác (OD).*
> - *Thứ hai là bộ dữ liệu **DRIVE** gồm 40 ảnh phục vụ phân vùng cây mao mạch võng mạc trong trường thị giác FOV.*
> 
> *Về mặt kiến trúc học sâu:*
> - *Chúng em đã xây dựng mô hình **Baseline U-Net** và kiến trúc nâng cấp **Attention U-Net**.*
> - *Điểm đột phá của Attention U-Net là việc bổ sung các khối **Attention Gates (Cổng chú ý)** tại các đường kết nối tắt (Skip Connections). Tín hiệu ngữ cảnh toàn cục từ tầng sâu của Decoder sẽ được dùng để lọc và nhân trọng số với các đặc trưng từ Encoder. Nhờ đó, các vùng nền võng mạc bình thường sẽ bị triệt tiêu, trong khi các đốm tổn thương vi phình mạch MA kích thước nhỏ sẽ được khuếch đại tín hiệu.*
> 
> *Về hàm mất mát:*
> - *Do diện tích tổn thương thường chiếm dưới 1% bề mặt võng mạc, xảy ra hiện tượng mất cân bằng dữ liệu cực kỳ nghiêm trọng. Chúng em đã áp dụng hàm mất mát kết hợp **BCEDiceLoss**, phối hợp độ mượt gradient của Binary Cross Entropy với khả năng tối ưu diện tích giao thoa của Soft Dice Loss.*
> 
> *Về giải thuật định lượng lâm sàng (Medical Evaluator):*
> - *Chúng em không chỉ dừng lại ở việc đếm pixel, mà sử dụng thuật toán **Connected Components Analysis (8-connectivity)** để đếm chính xác số lượng ổ tổn thương độc lập, đồng thời tự động bóc tách vùng trường thị giác hợp lệ (FOV Mask) để tính toán tỷ lệ phần trăm diện tích tổn thương trên bề mặt võng mạc thực tế."*

---

## 3. KIẾN TRÚC HỆ THỐNG VÀ CÔNG NGHỆ FULL-STACK (3.0 PHÚT)

> *"Thưa Hội đồng, để đưa mô hình AI vào vận hành thực tế, chúng em đã thiết kế hệ thống theo kiến trúc Microservices phân tán hiện đại:*
> - *Tầng Backend sử dụng **FastAPI**, xác thực bảo mật chuẩn **JWT Authentication** và phân quyền vai trò Doctor / Admin.*
> - *Cơ sở dữ liệu quan hệ **PostgreSQL** chuẩn hóa 3NF quản lý bệnh nhân, ca chụp, kết quả định lượng chi tiết cho từng loại tổn thương.*
> - *Hàng đợi tác vụ **Celery + Redis** đảm nhiệm việc xử lý bất đồng bộ, tách biệt hoàn toàn luồng giao tiếp Web khỏi luồng suy luận Deep Learning nặng, giúp hệ thống luôn phản hồi mượt mà và hỗ trợ cơ chế Polling trạng thái thời gian thực.*
> - *Hệ thống lưu trữ đối tượng **MinIO S3** kết hợp cơ chế Local Storage Fallback bảo đảm hệ thống có thể chạy ngoại tuyến mà không phụ thuộc hạ tầng đám mây.*
> - *Giao diện người dùng được xây dựng bằng **ReactJS, TypeScript, Tailwind CSS**, với điểm nhấn nổi bật là bộ quan sát ảnh **HTML5 Canvas Medical Viewer**.*
> - *Cuối cùng, hệ thống tích hợp thư viện **ReportLab** để tự động xuất bản phiếu kết quả xét nghiệm y khoa dạng PDF chuẩn quốc tế."*

---

## 4. KẾT QUẢ THỰC NGHIỆM VÀ DEMO SẢN PHẨM (4.0 PHÚT)

> *(Bắt đầu mở giao diện Demo hệ thống)*
> 
> *"Sau đây, em xin phép được trình diễn trực tiếp hệ thống đang chạy thực tế trên máy tính cá nhân có hỗ trợ GPU NVIDIA RTX 4050:*
> 
> 1. *Đầu tiên, tại màn hình Đăng nhập, hệ thống hỗ trợ đăng nhập bác sĩ hoặc quản trị viên. Chúng em có sẵn nút nạp tài khoản mẫu kiểm thử `admin / admin123`.*
> 2. *Vào màn hình **Dashboard**, các Thầy Cô có thể thấy bảng thống kê tổng thể: tổng số bệnh nhân, số ca chụp, biểu đồ phân bố số lượng ổ tổn thương (MA, HE, EX, SE, OD, Vessel) và bảng so sánh hiệu năng thực nghiệm giữa Baseline U-Net và Attention U-Net.*
> 3. *Chuyển sang phân hệ **Quản lý Bệnh nhân**, chúng ta có thể tạo mới hồ sơ bệnh án với mã tự sinh `PAT-2026-0001`, nhập tiền sử bệnh lý tiểu đường và quản lý toàn bộ lịch sử thăm khám.*
> 4. *Tiếp theo là phân hệ trung tâm: **Phòng Phân tích Y sinh (Analysis Studio)**:*
>    - *Tại đây, bác sĩ có thể tải ảnh chụp đáy mắt từ máy tính, hoặc chọn ngay các mẫu benchmark thực tế từ tập test của IDRiD và DRIVE.*
>    - *Chúng ta chọn nhiệm vụ IDRiD Lesions và mô hình Attention U-Net, sau đó bấm 'Bắt đầu Phân tích AI'.*
>    - *Tác vụ được đẩy vào hàng đợi Celery, thanh tiến trình hiển thị trạng thái PENDING -> PROCESSING -> COMPLETED.*
>    - *Khi hoàn tất, bộ quan sát **Canvas Viewer** hiển thị kết quả trực quan:*
>      - *Chúng ta có thể dùng con lăn chuột để Zoom mượt mà vào chi tiết tổn thương.*
>      - *Kéo chuột để di chuyển (Pan).*
>      - *Kéo thanh trượt **Split Slider** để so sánh trực tiếp trước và sau khi AI phân vùng.*
>      - *Bật/tắt độc lập các lớp nhãn tổn thương: Vi phình mạch màu đỏ, Xuất huyết màu cam, Xuất tiết cứng màu vàng, Đĩa thị màu xanh dương.*
>      - *Điều chỉnh thanh trượt **Opacity** từ 0% đến 100%.*
>    - *Bên dưới là Bảng định lượng lâm sàng: đếm được chính xác số lượng ổ tổn thương cho từng loại, diện tích pixel và tỷ lệ % trên trường thị giác FOV.*
> 5. *Cuối cùng, chỉ với 1 cú click chuột vào nút **'Xuất Báo cáo Y khoa (PDF)'**, hệ thống ReportLab sẽ biên dịch và tải ngay về file PDF phiếu kết quả xét nghiệm y khoa khổ A4 chuyên nghiệp, có đầy đủ logo VKU, thông tin bệnh nhân, ảnh đối sánh song song và cảnh báo y đức pháp lý.*
> 
> *Đặc biệt, hệ thống đã vượt qua toàn bộ **14/14 bài kiểm thử tự động bằng Pytest**, từ kiểm tra kiến trúc mạng, kiểm tra tránh rò rỉ dữ liệu (Data Leakage), giải thuật Connected Components cho đến toàn bộ API Backend."*

---

## 5. KẾT LUẬN VÀ LỜI CẢM ƠN (1.0 PHÚT)

> *"Thưa quý Thầy Cô trong Hội đồng,*
> 
> *Đồ án của chúng em đã hiện thực hóa một giải pháp toàn diện từ bài toán Deep Learning y sinh chuyên sâu đến một nền tảng phần mềm hoàn chỉnh, có khả năng ứng dụng thực tế hỗ trợ đắc lực cho công tác sàng lọc bệnh võng mạc tiểu đường.*
> 
> *Chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến các Thầy Cô trong Khoa Khoa học Máy tính và Hội đồng đã tận tình hướng dẫn, theo dõi và đóng góp ý kiến cho đề tài.*
> 
> *Em xin kết thúc bài báo cáo và kính mời quý Thầy Cô trong Hội đồng đặt câu hỏi phản biện ạ. Em xin trân trọng cảm ơn!"*

