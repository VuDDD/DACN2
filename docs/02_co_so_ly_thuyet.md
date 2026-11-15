# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ

**Đồ án chuyên ngành 2 – Trường Đại học Công nghệ Thông tin và Truyền thông Việt - Hàn (VKU)**  
**Đề tài:** Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng mô hình học sâu (Deep Learning)  

---

## 2.1. HỌC SÂU VÀ PHÂN VÙNG ẢNH Y TẾ (DEEP LEARNING & MEDICAL SEGMENTATION)

### 2.1.1. Tổng quan về Mạng Nơ-ron Tích chập (Convolutional Neural Networks - CNN)
Mạng nơ-ron tích chập (CNN) là kiến trúc học sâu chủ đạo trong lĩnh vực thị giác máy tính nhờ hai đặc tính cốt lõi:
1. **Chia sẻ trọng số (Weight Sharing):** Các bộ lọc (filters/kernels) quét qua toàn bộ bức ảnh giúp giảm đáng kể số lượng tham số huấn luyện so với mạng Fully Connected truyền thống.
2. **Bất biến với phép tịnh tiến (Translation Invariance):** Cho phép phát hiện các đặc trưng cục bộ (cạnh, góc, kết cấu) tại bất kỳ vị trí nào trong không gian ảnh võng mạc.

### 2.1.2. Bài toán Phân vùng Ngữ nghĩa (Semantic Segmentation)
Trong khi bài toán Phân loại (Classification) gán nhãn cho toàn bộ bức ảnh, bài toán Phân vùng Ngữ nghĩa (Semantic Segmentation) thực hiện dự đoán phân loại tại từng điểm ảnh (Pixel-wise Classification). Đối với một ảnh đầu vào $I \in \mathbb{R}^{H \times W \times C}$, mô hình phân vùng dự đoán một ma trận xác suất $\hat{Y} \in [0, 1]^{H \times W \times K}$, trong đó $K$ là số lượng lớp bệnh học cần nhận diện.

Trong bối cảnh ảnh đáy mắt võng mạc:
- Bài toán IDRiD là bài toán phân vùng đa nhãn (Multi-label Segmentation) với $K=5$ (MA, HE, EX, SE, OD), trong đó một điểm ảnh có thể đồng thời thuộc ranh giới của nhiều đặc trưng (ví dụ: xuất huyết bao quanh vi phình mạch).
- Bài toán DRIVE là bài toán phân vùng nhị phân (Binary Segmentation) với $K=1$, xác định ranh giới cây mao mạch nằm trong vùng trường thị giác (Field of View - FOV).

---

## 2.2. KIẾN TRÚC MÔ HÌNH U-NET

Kiến trúc U-Net được Ronneberger và các cộng sự giới thiệu tại hội nghị MICCAI 2015, nhanh chóng trở thành tiêu chuẩn vàng trong xử lý ảnh y sinh nhờ khả năng huấn luyện hiệu quả ngay cả khi tập dữ liệu có số lượng mẫu giới hạn.

```
       Input Image [3, H, W]
               |
        [Double Conv 64] ---------------- Skip 1 ---------------> [Double Conv 64] -> Output [K, H, W]
               v                                                         ^
         [MaxPool 2x2]                                            [UpConv 128->64]
               v                                                         |
        [Double Conv 128] --------------- Skip 2 ---------------> [Double Conv 128]
               v                                                         ^
         [MaxPool 2x2]                                            [UpConv 256->128]
               v                                                         |
        [Double Conv 256] --------------- Skip 3 ---------------> [Double Conv 256]
               v                                                         ^
         [MaxPool 2x2]                                            [UpConv 512->256]
               v                                                         |
        [Double Conv 512] --------------- Skip 4 ---------------> [Double Conv 512]
               v                                                         ^
         [MaxPool 2x2]                                            [UpConv 1024->512]
               v                                                         |
        [Double Conv 1024] (Bottleneck / Deep Latent Representation) -----+
```

### Các thành phần chính của U-Net:
1. **Nhánh co rút (Encoder / Contracting Path):** Bao gồm các khối Double Convolution (Conv $3 \times 3 \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Conv} 3 \times 3 \rightarrow \text{BatchNorm} \rightarrow \text{ReLU}$) theo sau bởi phép gộp cực đại (MaxPooling $2 \times 2$) để giảm độ phân giải không gian và trích xuất các đặc trưng ngữ nghĩa mức cao (High-level semantic features).
2. **Nút thắt cổ chai (Bottleneck):** Nằm ở tầng sâu nhất, nén bức ảnh thành biểu diễn tiềm ẩn cô đọng mang tính ngữ cảnh toàn cục.
3. **Nhánh mở rộng (Decoder / Expanding Path):** Sử dụng phép tích chập chuyển vị (Transposed Convolution) để khôi phục kích thước không gian nguyên bản của ảnh.
4. **Kết nối tắt (Skip Connections):** Đây là sáng tạo then chốt của U-Net. Các bản đồ đặc trưng có độ phân giải cao từ tầng Encoder được nối trực tiếp (concatenation) với các bản đồ đặc trưng ở tầng Decoder tương ứng. Cơ chế này bù đắp lượng thông tin không gian bị mất trong quá trình MaxPooling, giúp mô hình tái tạo các ranh giới tổn thương sắc nét.

---

## 2.3. KIẾN TRÚC ATTENTION U-NET VÀ CƠ CHẾ CỔNG CHÚ Ý (ATTENTION GATES)

### 2.3.1. Hạn chế của U-Net truyền thống
Trong U-Net tiêu chuẩn, các kết nối tắt (Skip Connections) chuyển tiếp toàn bộ thông tin từ Encoder sang Decoder mà không có cơ chế chọn lọc. Điều này dẫn đến việc chuyển giao cả các đặc trưng nhiễu, các vùng nền võng mạc bình thường không liên quan, làm giảm độ tập trung của Decoder vào các tổn thương vi mô cực nhỏ như Microaneurysms (MA).

### 2.3.2. Nguyên lý Cổng Chú ý (Attention Gate - AG)
Oktay và các cộng sự (2018) đã đề xuất kiến trúc **Attention U-Net** bằng cách tích hợp mô-đun Attention Gate (AG) tại mỗi nhánh kết nối tắt trước khi thực hiện ghép nối đặc trưng.

```
       Skip Connection x (từ Encoder) ---------> [ Conv 1x1 ] ---+
                                                                  |
                                                                  v
                                                               (+) ---> [ ReLU ] ---> [ Conv 1x1 ] ---> [ Sigmoid ]
                                                                  ^                                           |
                                                                  |                                           v
       Gating Signal g (từ Decoder sâu hơn) ---> [ Conv 1x1 ] ----+                                     Attention Coefficients psi
                                                                                                              |
                                                                                                              v
       Output Features = x * psi <----------------------------------------------------------------------------+
```

Công thức toán học của Attention Gate:
$$\alpha = \sigma_2 \left( \psi^T \left( \sigma_1 \left( W_x^T x_l + W_g^T g + b_g \right) \right) + b_\psi \right)$$
$$\hat{x}_l = x_l \cdot \alpha$$

Trong đó:
- $x_l$ là bản đồ đặc trưng từ tầng kết nối tắt $l$ của Encoder.
- $g$ là tín hiệu cổng (Gating Signal) lấy từ tầng Decoder sâu hơn, mang thông tin ngữ cảnh toàn cục.
- $W_x, W_g, \psi$ là các phép biến đổi tuyến tính thực hiện bằng tích chập $1 \times 1$.
- $\sigma_1$ là hàm kích hoạt ReLU, $\sigma_2$ là hàm Sigmoid chuẩn hóa hệ số chú ý $\alpha \in [0, 1]$.
- $\hat{x}_l$ là bản đồ đặc trưng đã được lọc: các pixel thuộc vùng tổn thương bệnh học có hệ số $\alpha \to 1$ sẽ được giữ lại trọn vẹn, trong khi các pixel nền có $\alpha \to 0$ sẽ bị ức chế.

---

## 2.4. HÀM MẤT MÁT TRONG PHÂN VÙNG ẢNH Y SINH (LOSS FUNCTIONS)

Trong bài toán phân vùng tổn thương võng mạc, hiện tượng mất cân bằng mẫu (Class Imbalance) diễn ra cực kỳ trầm trọng: diện tích của các ổ vi phình mạch (MA) hoặc xuất tiết (EX) thường chỉ chiếm dưới **0.1% đến 1.5%** tổng diện tích đáy mắt; hơn 98.5% còn lại là nền võng mạc bình thường. Nếu chỉ sử dụng hàm Binary Cross Entropy (BCE) thông thường, mô hình sẽ dễ dàng bị tối ưu hóa cục bộ theo hướng dự đoán toàn bộ pixel là nền.

### 2.4.1. Binary Cross Entropy Loss (BCE)
$$\mathcal{L}_{BCE}(y, p) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$
Ưu điểm: Cung cấp gradient mượt mà và ổn định ở các giai đoạn đầu huấn luyện.

### 2.4.2. Soft Dice Loss
Được dẫn xuất trực tiếp từ hệ số tương đồng Dice-Sørensen:
$$\mathcal{L}_{Dice}(y, p) = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$
Trong đó $\epsilon = 10^{-5}$ là hệ số làm mịn (smoothing factor) ngăn chặn lỗi chia cho 0 và ổn định gradient khi mặt nạ ground truth rỗng.

### 2.4.3. Hàm mất mát kết hợp BCEDiceLoss
Đồ án áp dụng kết hợp có trọng số giữa BCE và Dice Loss:
$$\mathcal{L}_{Total} = w_{bce} \cdot \mathcal{L}_{BCE} + w_{dice} \cdot \mathcal{L}_{Dice}$$
Với $w_{bce} = 0.5$ và $w_{dice} = 0.5$. Sự phối hợp này bảo đảm mô hình vừa học tốt độ dốc xác suất tại từng điểm ảnh (BCE), vừa tối ưu hóa chỉ số diện tích giao thoa tổng thể (Dice), khắc phục triệt để bài toán mất cân bằng dữ liệu.

---

## 2.5. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ Y SINH (EVALUATION METRICS)

Để đánh giá khách quan hiệu năng của mô hình so với Ground Truth chuẩn của chuyên gia, hệ thống tính toán 6 chỉ số tiêu chuẩn:

1. **Dice Similarity Coefficient (F1-Score):**
   $$\text{Dice} = \frac{2 \times |P \cap G|}{|P| + |G|} = \frac{2 TP}{2 TP + FP + FN}$$
2. **Intersection over Union (IoU / Jaccard Index):**
   $$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}$$
3. **Precision (Độ chuẩn xác / Giá trị dự đoán dương):**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
4. **Recall / Sensitivity (Độ nhạy):** Khả năng phát hiện không bỏ sót tổn thương:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
5. **Specificity (Độ đặc hiệu):** Khả năng nhận diện chính xác các vùng võng mạc khỏe mạnh:
   $$\text{Specificity} = \frac{TN}{TN + FP}$$

---

## 2.6. GIẢI THUẬT ĐỊNH LƯỢNG LÂM SÀNG (CONNECTED COMPONENTS & FOV QUANTIFICATION)

### 2.6.1. Phân tích Thành phần Liên thông (Connected Components Analysis - CCA)
Nhằm cung cấp thông tin số lượng ổ tổn thương (Lesion Foci Count) thay vì chỉ báo cáo số pixel trừu tượng, hệ thống áp dụng giải thuật Connected Components với lân cận 8 điểm (8-connectivity) dựa trên thư viện OpenCV:
- Quét qua mặt nạ nhị phân dự đoán $M$.
- Đánh nhãn các miền liên thông độc lập $L_1, L_2, \dots, L_m$.
- Lọc nhiễu kích thước 1-pixel do cảm biến quang học máy ảnh, bảo toàn nghiêm ngặt các tổn thương có diện tích $\ge 2$ pixel.

### 2.6.2. Đo lường Diện tích trên Trường Thị giác Hợp lệ (Retinal FOV Ratio)
Mỗi ảnh đáy mắt có diện tích vòng tròn trường thị giác khác nhau phụ thuộc góc mở ống kính (Field of View - $45^\circ$ hoặc $50^\circ$). Vùng màu đen bên ngoài ống kính không phải là mô võng mạc sinh học. Do đó, công thức định lượng tỷ lệ diện tích tổn thương bắt buộc phải chuẩn hóa theo vùng FOV hợp lệ:
$$\text{Lesion Percentage (\%)} = \left( \frac{\text{Lesion Pixels}}{\text{Valid Retinal FOV Pixels}} \right) \times 100\%$$
Quy tắc kỹ thuật: Chỉ số được hiển thị minh bạch bằng đơn vị pixel và tỷ lệ \% FOV; hệ thống tuyệt đối không tự ý chuyển đổi sang $\text{mm}^2$ khi chưa có thông số hiệu chuẩn quang học của thiết bị chụp đáy mắt.

---

## 2.7. HỆ THỐNG CÔNG NGHỆ NỀN TẢNG (TECHNOLOGY STACK)

1. **Backend Service:** **Python 3.11 / 3.13 + FastAPI** (hiệu năng cao, async/await native, tự động sinh tài liệu Swagger OpenAPI UI, Pydantic V2 validation).
2. **Database:** **PostgreSQL + SQLAlchemy 2.0 ORM**, kết hợp cơ chế linh hoạt chuyển đổi SQLite cho môi trường máy trạm độc lập.
3. **Queue & Async Worker:** **Celery 5.3 + Redis 7**, xử lý luồng suy luận Deep Learning bất đồng bộ, cập nhật tiến trình phân tích theo thời gian thực (Polling / Status endpoint).
4. **Storage:** **MinIO Object Storage** tương thích chuẩn Amazon S3 API, kèm lớp trừu tượng lưu trữ nội bộ (Local Fallback Storage).
5. **Frontend:** **React 18 + Vite + TypeScript + Tailwind CSS**, tích hợp **HTML5 Canvas API** xây dựng bộ quan sát ảnh đáy mắt y khoa đa lớp mượt mà, **Recharts** vẽ biểu đồ thống kê.
6. **Báo cáo:** **ReportLab 4.1**, sinh phiếu kết quả xét nghiệm y khoa dạng PDF chuẩn quốc tế.
7. **Đóng gói:** **Docker & Docker Compose**, hỗ trợ cấu hình đa dịch vụ với chế độ CPU và GPU CUDA.

