# TÀI LIỆU CÂU HỎI PHẢN BIỆN VÀ TRẢ LỜI BẢO VỆ ĐỒ ÁN (VIVA DEFENSE Q&A)

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  

---

### Câu hỏi 1: Tại sao nhóm lại lựa chọn bài toán Phân vùng tổn thương (Semantic Segmentation) thay vì bài toán Phân loại (Classification) vốn đơn giản hơn?
**Trả lời:**
- Trong thực hành lâm sàng nhãn khoa, một kết luận phân loại nhị phân (ví dụ: "Có bệnh võng mạc tiểu đường") chỉ mang tính cảnh báo sơ khởi, không đủ thông tin cho bác sĩ đưa ra quyết định can thiệp điều trị.
- Bác sĩ chuyên khoa mắt cần biết:
  1. **Hình thái và bản chất tổn thương:** Đó là vi phình mạch (MA), xuất huyết (HE) hay xuất tiết (EX)?
  2. **Vị trí không gian:** Tổn thương nằm ở ngoại vi hay đang đe dọa vùng hoàng điểm (Macula)?
  3. **Định lượng lâm sàng:** Có bao nhiêu ổ tổn thương và chiếm bao nhiêu diện tích trên trường thị giác?
- Do đó, bài toán Phân vùng ở cấp độ điểm ảnh (Pixel-level Segmentation) là con đường duy nhất cung cấp được ranh giới chính xác để trích xuất các chỉ số định lượng khách quan phục vụ theo dõi tiến trình điều trị.

---

### Câu hỏi 2: Cơ chế Cổng Chú ý (Attention Gate - AG) trong Attention U-Net hoạt động như thế nào và tại sao nó lại giúp phát hiện tổn thương vi mô tốt hơn U-Net truyền thống?
**Trả lời:**
- Trong U-Net truyền thống, các đường kết nối tắt (Skip Connections) chuyển tiếp trực tiếp toàn bộ bản đồ đặc trưng từ tầng Encoder sang Decoder. Quá trình này vô tình chuyển giao cả các đặc trưng nhiễu, các vùng nền võng mạc khỏe mạnh không có bệnh lý.
- Attention Gate khắc phục nhược điểm này bằng cách sử dụng tín hiệu cổng $g$ (lấy từ tầng Decoder sâu hơn, vốn chứa thông tin ngữ cảnh toàn cục) để điều biến các đặc trưng $x_l$ từ Encoder.
- Thông qua phép tính cộng tuyến tính, kích hoạt ReLU, tích chập $1 \times 1$ và hàm Sigmoid, mô hình sinh ra một ma trận trọng số chú ý không gian $\alpha \in [0, 1]$.
- Các pixel thuộc vùng tổn thương bệnh lý (như đốm đỏ vi phình mạch MA) sẽ có hệ số $\alpha \to 1$ và được truyền tiếp trọn vẹn, trong khi các vùng nền bình thường có $\alpha \to 0$ sẽ bị triệt tiêu gradient. Nhờ đó, Decoder tập trung tối đa vào các ổ tổn thương nhỏ và hạn chế dương tính giả.

---

### Câu hỏi 3: Trong ảnh đáy mắt, diện tích tổn thương chỉ chiếm dưới 1% bề mặt võng mạc. Nhóm giải quyết bài toán mất cân bằng dữ liệu (Class Imbalance) cực độ này như thế nào?
**Trả lời:**
- Chúng em áp dụng chiến lược đa tầng:
  1. **Hàm mất mát kết hợp `BCEDiceLoss`:** Nếu chỉ dùng BCE thông thường, mô hình sẽ bị tối ưu hóa theo hướng dự đoán toàn bộ là nền (Background). Chúng em kết hợp Soft Dice Loss ($w_{dice}=0.5$) - hàm tối ưu trực tiếp hệ số tương giao diện tích (Intersection over Union), buộc mô hình phải phạt rất nặng nếu bỏ sót các pixel tổn thương nhỏ.
  2. **Tăng cường dữ liệu có chọn lọc (Selective Augmentation):** Áp dụng phép lật, xoay hình học đồng bộ, kết hợp điều chỉnh tương phản CLAHE trên kênh Green để làm nổi bật đặc trưng tổn thương trước khi đưa vào mạng.

---

### Câu hỏi 4: Làm thế nào nhóm bảo đảm không có hiện tượng rò rỉ dữ liệu (Data Leakage) giữa tập Huấn luyện (Train), Thẩm định (Validation) và Kiểm thử (Test)?
**Trả lời:**
- Chúng em tuân thủ nghiêm ngặt nguyên tắc phân chia ở cấp độ ảnh gốc bệnh nhân (Patient/Image-level split):
  - Bộ dữ liệu IDRiD có sự phân chia chính thức từ ban tổ chức: 54 ảnh Train và 27 ảnh Test.
  - Tập 54 ảnh Train được chúng em chia tách thành Train (46 ảnh) và Validation (8 ảnh) bằng bộ tạo số ngẫu nhiên cố định (`seed=42`).
  - **Quy tắc tuyệt đối:** Các phép tăng cường dữ liệu (Augmentation) chỉ được áp dụng trong lúc huấn luyện (online batch training). Không có bất kỳ ảnh gốc hoặc bản sao tăng cường nào của cùng một ảnh xuất hiện đồng thời ở nhiều tập.
  - Toàn vẹn này đã được kiểm chứng tự động và vượt qua bài test `test_idrid_dataset_leakage_and_shapes` trong Pytest.

---

### Câu hỏi 5: Tại sao hệ thống lại tách riêng quy trình huấn luyện bộ dữ liệu IDRiD và DRIVE mà không gộp chung vào một mô hình?
**Trả lời:**
- Về bản chất y sinh học, hai bộ dữ liệu giải quyết hai nhiệm vụ hoàn toàn khác nhau:
  - **IDRiD:** Nhận diện 5 dạng tổn thương bệnh lý vi mạch đái tháo đường và cấu trúc đĩa thị (MA, HE, EX, SE, OD).
  - **DRIVE:** Nhận diện toàn bộ cây giải phẫu mao mạch bình thường và bệnh lý nằm trong trường thị giác FOV.
- Việc gộp chung các mặt nạ khác bản chất sinh học thành một nhãn chung sẽ gây xung đột biểu diễn ngữ nghĩa (Semantic Ambiguity), làm giảm độ chính xác của mạng. Do đó, chúng em thiết kế hai nhánh AI chuyên biệt (Dual-Branch AI), mỗi nhánh sở hữu bộ trọng số tối ưu hóa riêng biệt cho nhiệm vụ của mình.

---

### Câu hỏi 6: Giải thuật đếm số ổ tổn thương và tính tỷ lệ phần trăm diện tích trên FOV hoạt động như thế nào?
**Trả lời:**
- **Đếm số ổ tổn thương:** Mô hình không chỉ đếm số pixel đơn lẻ, mà áp dụng giải thuật Connected Components Analysis (lân cận 8 điểm). Thuật toán quét và nhóm các pixel dương tính tiếp giáp nhau thành các miền liên thông độc lập (foci), đồng thời lọc nhiễu các điểm ảnh cô lập kích thước $< 2$ pixel do cảm biến camera.
- **Tính diện tích trên FOV:** Vùng màu đen bao quanh ảnh chụp đáy mắt là khoảng tối bên ngoài ống kính camera, không phải mô sinh học. Do đó, hệ thống trích xuất mặt nạ trường thị giác tròn (FOV Mask). Tỷ lệ phần trăm tổn thương được tính bằng:
  $$\text{Tỷ lệ (\%) } = \left( \frac{\text{Số pixel tổn thương}}{\text{Số pixel thuộc vùng FOV hợp lệ}} \right) \times 100\%$$
- Điều này bảo đảm tính khách quan và có thể so sánh giữa các góc mở camera khác nhau ($45^\circ$ hay $50^\circ$).

---

### Câu hỏi 7: Tại sao lại áp dụng hàng đợi bất đồng bộ Celery + Redis thay vì gọi trực tiếp hàm suy luận AI trong API endpoint của FastAPI?
**Trả lời:**
- Quá trình tiền xử lý, chạy mạng học sâu và tính toán Connected Components có thể mất từ vài trăm mili-giây đến vài giây tùy thuộc phần cứng GPU/CPU.
- Nếu gọi đồng bộ trong luồng xử lý Web chính:
  1. Sẽ gây nghẽn luồng (Blocking Event Loop), khiến máy chủ không thể phản hồi các yêu cầu khác từ người dùng.
  2. Dễ dẫn đến lỗi HTTP 504 Gateway Timeout khi có nhiều bác sĩ cùng tải ảnh lên phân tích.
- Bằng cách sử dụng **Celery + Redis**, tác vụ được đẩy vào hàng đợi bất đồng bộ, API lập tức phản hồi mã xác nhận HTTP 202 kèm `analysis_id`. Phía Client sử dụng cơ chế Polling nhẹ nhàng để theo dõi trạng thái `PENDING -> PROCESSING -> COMPLETED`, mang lại trải nghiệm người dùng chuyên nghiệp và khả năng mở rộng (Scalability) linh hoạt.

---

### Câu hỏi 8: Giá trị pháp lý của Báo cáo Y khoa PDF là gì và hệ thống xử lý vấn đề Y đức (Medical AI Ethics) như thế nào?
**Trả lời:**
- Đồ án quán triệt rõ ràng: **Hệ thống là công cụ hỗ trợ quyết định lâm sàng (Clinical Decision Support System - CDSS), tuyệt đối không tự động kết luận bệnh hay thay thế bác sĩ.**
- Trên mỗi phiếu báo cáo PDF kết xuất tự động bởi thư viện ReportLab:
  1. Chỉ số diện tích hiển thị rõ đơn vị pixel và tỷ lệ \% FOV, không tự ý quy đổi ra $\text{mm}^2$ khi chưa có hiệu chuẩn quang học thiết bị.
  2. Bắt buộc có chữ ký xác nhận của Bác sĩ nhãn khoa phụ trách (Consulting Physician Signature).
  3. Luôn có điều khoản cảnh báo y đức pháp lý (Legal Medical Disclaimer) in rõ ở chân trang, khẳng định kết quả AI mang tính chất tham khảo nghiên cứu và hỗ trợ chẩn đoán hình ảnh.

