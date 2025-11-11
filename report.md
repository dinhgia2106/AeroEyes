# BÁO CÁO KỸ THUẬT: PIPELINE TUYỆT ĐỐI CHO BÀI TOÁN AEROEYES (ZALO AI CHALLENGE 2025)

## PHẦN 1: TÓM TẮT QUẢN TRỊ VÀ PHÂN TÍCH CHIẾN LƯỢC

### 1.1 Mục tiêu Báo cáo

Báo cáo này trình bày một pipeline kiến trúc và triển khai end-to-end (toàn diện) chi tiết để giải quyết "Track 1: Drones for flood victim search" của Zalo AI Challenge 2025 (sau đây gọi là "AeroEyes"). Mục tiêu chiến lược là tối đa hóa chỉ số đánh giá chính, STIoU (Spatio-Temporal Intersection over Union), đồng thời tuân thủ một cách nghiêm ngặt ràng buộc vận hành quan trọng nhất: duy trì hiệu suất suy luận (inference performance) thời gian thực > 15 FPS (Frames Per Second) trên nền tảng phần cứng NVIDIA Jetson.

### 1.2 Tuyên bố Chiến lược Cốt lõi

Một cách tiếp cận tiêu chuẩn, chẳng hạn như sử dụng một bộ phát hiện đối tượng (object detector) trên từng khung hình (per-frame) và hy vọng nó hoạt động, gần như chắc chắn sẽ thất bại. Bài toán này có các ràng buộc mâu thuẫn lẫn nhau (hiệu suất thời gian thực so với phát hiện vật thể nhỏ) và một chỉ số đánh giá phi tiêu chuẩn (STIoU). Chiến lược chiến thắng đòi hỏi một giải pháp hybrid (lai) tích hợp, được thiết kế tùy chỉnh, bao gồm 4 giai đoạn, giải quyết từng thách thức một cách có chủ đích:

*   **Pipeline Dữ liệu (Augmentation):** Triển khai một chiến lược tăng cường dữ liệu (data augmentation) mạnh mẽ, tập trung vào việc bù đắp cho các ràng buộc về hiệu suất (performance) bằng cách giải quyết các vật thể nhỏ và khoảng cách miền (domain gap) trong giai đoạn huấn luyện.
*   **Kiến trúc Mô hình (Architecture):** Xây dựng một kiến trúc "Siamese-YOLO" tùy chỉnh, nhẹ (lightweight), để thực hiện phát hiện dựa trên tham chiếu (reference-based detection) một cách hiệu quả.
*   **Hậu kỳ (Post-Processing):** Sử dụng một pipeline xử lý hậu kỳ dựa trên theo dõi (tracking-based), được tối ưu hóa đặc biệt để tối đa hóa chỉ số STIoU bằng cách thực thi sự nhất quán về mặt thời gian (temporal consistency).
*   **Triển khai (Acceleration):** Tối ưu hóa toàn bộ pipeline bằng cách sử dụng các công cụ tăng tốc phần cứng chuyên dụng như NVIDIA TensorRT và DeepStream để đáp ứng yêu cầu > 15 FPS.

### 1.3 Các Quyết định Kiến trúc then chốt

Việc lựa chọn chiến lược cốt lõi được định hình bởi việc loại bỏ một cách có ý thức các phương pháp phổ biến nhưng không phù hợp:

*   **Loại bỏ SAHI (Slicing Aided Hyper Inference):** Mặc dù các kỹ thuật "tiling" (chia nhỏ ảnh) như SAHI là phương pháp hiện đại (SOTA) để phát hiện các vật thể siêu nhỏ, chúng đòi hỏi chi phí tính toán cực kỳ cao (thực hiện suy luận nhiều lần trên mỗi khung hình). Phân tích cho thấy SAHI thậm chí có thể chạy ở ≈ 1 FPS trên các GPU máy trạm cao cấp, khiến nó không thể nào đạt được > 15 FPS trên một thiết bị biên (edge device) như Jetson.
*   **Loại bỏ Tracker thuần túy (SiamRPN):** Các mô hình Theo dõi Đối tượng Đơn (Single Object Tracker - SOT) tiêu chuẩn như SiamRPN++ có vẻ phù hợp về mặt kiến trúc (Siamese), nhưng chúng được thiết kế để theo dõi (track) một đối tượng sau khi được cung cấp một bounding box ở khung hình đầu tiên. Bài toán của chúng ta là phát hiện (detect) vật thể trong toàn bộ video, không phải theo dõi nó từ một điểm bắt đầu đã biết.
*   **Lựa chọn: Bộ phát hiện Hybrid (Query-YOLO):** Con đường khả thi duy nhất là một giải pháp hybrid. Chúng ta sẽ lấy một bộ phát hiện đối tượng thời gian thực (real-time detector) đã được chứng minh là cực kỳ nhanh (ví dụ: YOLOv8) và sửa đổi (modify) nó để trở thành một bộ phát hiện dựa trên truy vấn (query-based detector). Cách tiếp cận này cân bằng giữa tốc độ thô (raw speed) của YOLO và yêu cầu học theo mẫu (matching) của bài toán.

## PHẦN 2: GIẢI MÃ THỬ THÁCH: CÁC RÀNG BUỘC MÂU THUẪN CỦA AEROEYES

Một phân tích sâu về các yêu cầu của cuộc thi cho thấy bốn ràng buộc cốt lõi, trong đó nhiều ràng buộc mâu thuẫn trực tiếp với nhau. Một pipeline thành công phải giải quyết và cân bằng tất cả các mâu thuẫn này.

### 2.1 Ràng buộc 1: Hiệu suất Jetson >15 FPS (Ràng buộc Cứng)

Đây là ràng buộc nghiêm ngặt và mang tính quyết định nhất của toàn bộ bài toán.

*   **Phân tích:** Yêu cầu > 15 FPS tương đương với ngân sách thời gian tối đa là 66.67 mili giây (ms) cho mỗi khung hình video. Điều quan trọng là ngân sách này bao gồm toàn bộ pipeline: giải mã (decode) khung hình video, tiền xử lý (pre-processing), suy luận mô hình (model inference), và xử lý hậu kỳ (post-processing).
*   **Bối cảnh:** Các benchmark (kiểm chuẩn) trên nền tảng Jetson (bao gồm Nano, TX2, Xavier, và Orin) cho thấy ngay cả các mô hình được tối ưu hóa cao cũng gặp khó khăn để đạt được tốc độ khung hình cao một cách nhất quán. Một mô hình ResNet-10 có thể chạy nhanh, nhưng các kiến trúc phức tạp hơn như YOLOv8n có thể chỉ đạt được 12-15 FPS khi chạy trong một pipeline Python đơn giản, chưa tối ưu hóa.
*   **Tác động:** Điều này có nghĩa là bản thân mô hình suy luận (phần model.predict()) phải chạy nhanh hơn đáng kể so với 66ms, lý tưởng là dưới 30-40ms, để chừa không gian cho các chi phí I/O và xử lý hậu kỳ. Sự thật này ngay lập tức loại bỏ mọi kiến trúc backbone lớn hơn YOLOv8n, EfficientNet-B0, hoặc RT-DETR-R18. Bất kỳ giải pháp nào liên quan đến các mô hình Transformer lớn hoặc các backbone nặng (heavy backbones) đều bị loại bỏ ngay từ đầu.

### 2.2 Ràng buộc 2: Vật thể Siêu nhỏ và Điều kiện Khắc nghiệt

Ràng buộc này mâu thuẫn trực tiếp với Ràng buộc 1.

*   **Phân tích:** Dữ liệu được cung cấp cho thấy các vật thể "rất nhỏ, khó nhìn, có thể bị che khuất". Các biểu đồ phân phối bounding box cho thấy tỷ lệ diện tích (Area Ratio) thường xuyên dưới 0.0005 (tức là 0.05% diện tích khung hình). Thêm vào đó, video được quay trong "điều kiện ánh sáng khắc nghiệt (nắng gắt, thiếu sáng,...)".
*   **Mâu thuẫn:** Các phương pháp tốt nhất để phát hiện vật thể nhỏ thường liên quan đến việc xử lý hình ảnh ở độ phân giải cao, sử dụng các Feature Pyramid Networks (FPN) phức tạp, hoặc áp dụng các kỹ thuật suy luận đa tỷ lệ (multi-scale inference) như SAHI. Tất cả những phương pháp này đều tốn kém về mặt tính toán và vi phạm trực tiếp Ràng buộc 1.
*   **Giải pháp Chiến lược:** Chúng ta không thể giải quyết vấn đề vật thể nhỏ vào lúc suy luận (inference-time) vì nó quá chậm. Do đó, chúng ta phải chuyển gánh nặng giải quyết vật thể nhỏ sang lúc huấn luyện (training-time). Pipeline phải bắt buộc mô hình học cách phát hiện các vật thể nhỏ trên toàn bộ khung hình, bằng cách sử dụng các kỹ thuật tăng cường dữ liệu mạnh mẽ như Tiling và Copy-Paste.

### 2.3 Ràng buộc 3: Khoảng cách Miền (Domain Gap) Ground-to-Aerial

Đây là thách thức cốt lõi về "metric learning" (học mêtric) của bài toán.

*   **Phân tích:** Hình ảnh tham chiếu là ảnh chụp từ mặt đất (ground-view), độ phân giải cao (ví dụ: 2922 x 2922), rõ nét, và được chụp ở góc nhìn ngang tầm mắt. Dữ liệu mục tiêu là video từ fly cam (aerial-view), góc nhìn từ trên xuống (top-down), mờ hơn, và vật thể xuất hiện ở một góc độ hoàn toàn khác.
*   **Mâu thuẫn:** Mô hình phải học cách nhận ra rằng "Backpack_0" (ảnh ground-view) và một mảng pixel mờ (ảnh aerial-view) là cùng một vật thể. Các phương pháp cố gắng "dịch" (translate) hình ảnh giữa các miền, ví dụ như sử dụng cGANs để tổng hợp một ảnh aerial-view từ một ảnh ground-view, là quá phức tạp và không thể chạy trong thời gian thực.
*   **Giải pháp Chiến lược:** Vấn đề không được giải quyết bằng cách thay đổi hình ảnh đầu vào, mà bằng cách định hình không gian đặc trưng (feature space). Giải pháp là huấn luyện một bộ trích xuất đặc trưng (feature extractor) duy nhất có khả năng ánh xạ cả hai miền (ground và aerial) vào cùng một điểm trong không gian embedding. Điều này trực tiếp dẫn đến việc sử dụng một hàm mất mát metric learning, cụ thể là Triplet Loss. Hàm loss này sẽ "kéo" biểu diễn của vật thể aerial (anchor) và biểu diễn của vật thể ground (positive) lại gần nhau, đồng thời "đẩy" chúng ra xa các vật thể không liên quan (negative).

### 2.4 Ràng buộc 4: Chỉ số Đánh giá STIoU (Spatio-Temporal IoU)

Đây là ràng buộc cuối cùng và tinh vi nhất, ảnh hưởng đến thiết kế xử lý hậu kỳ.

*   **Phân tích:** Công thức STIoU, $STIoU = \frac{\sum_{f \in intersection} IoU(B_f, B'_f)}{\sum_{f \in union} 1}$, không chỉ đơn giản là IoU trung bình. Nó là tổng của IoU trên các khung hình chung (intersection), chia cho tổng số lượng khung hình trong cả ground-truth và dự đoán (union).
*   **Các kịch bản thất bại:**
    *   **Phát hiện bị bỏ lỡ (False Negative):** Mô hình phát hiện vật thể ở khung f và f+2, nhưng bỏ lỡ ở khung f+1. Tử số (∑IoU) không đổi, nhưng Mẫu số (∑union) tăng thêm 1 (vì f+1 nằm trong ground-truth union). Điều này làm giảm điểm số tổng.
    *   **Phát hiện "giật" (Jitter):** Bounding box của mô hình dao động (flicker) hoặc "giật" (jitter) vài pixel giữa các khung hình. Điều này làm giảm nhẹ IoU ở mỗi khung hình, gây ra sự sụt giảm đáng kể cho Tử số (∑IoU) khi cộng dồn lại.
    *   **Phát hiện sai (False Positive):** Một dự đoán sai ngẫu nhiên ở một khung hình bất kỳ sẽ tăng Mẫu số lên 1 mà không tăng Tử số, làm giảm điểm số.
*   **Giải pháp Chiến lược:** Một mô hình phát hiện trên từng khung hình (per-frame) tiêu chuẩn như YOLO vốn dĩ không ổn định về mặt thời gian và dễ bị giật. Để tối đa hóa STIoU, pipeline của chúng ta phải thực thi sự nhất quán về mặt thời gian. Công nghệ được thiết kế chính xác cho việc này là Bộ lọc Kalman. Nó làm mượt các phép đo (detections) bị nhiễu và dự đoán (predict) trạng thái (vị trí) của vật thể khi phép đo bị mất (missed detection). Một triển khai (implementation) nhẹ, hiệu quả của bộ lọc Kalman cho việc theo dõi là thuật toán SORT (Simple Online and Realtime Tracking). Do đó, xử lý hậu kỳ với SORT/Kalman Filter không phải là một tùy chọn, mà là một thành phần bắt buộc của pipeline để "lấp đầy" các khung hình bị bỏ lỡ và "làm mượt" các bounding box bị giật, tối ưu hóa trực tiếp cho cả tử số và mẫu số của STIoU.

## PHẦN 3: TỔNG QUAN PIPELINE TUYỆT ĐỐI (A4: ANALYSIS, AUGMENTATION, ARCHITECTURE, ACCELERATION)

Dựa trên phân tích các ràng buộc mâu thuẫn, pipeline "tuyệt đối" được đề xuất là một quy trình 4 giai đoạn tích hợp. Mỗi giai đoạn được thiết kế để giải quyết một tập hợp các thách thức cụ thể:

*   **GIAI ĐOẠN 1: DỮ LIỆU (Preprocessing & Augmentation):** Chuẩn bị và tăng cường dữ liệu huấn luyện để giải quyết ba vấn đề chính: Rò rỉ Dữ liệu (Data Leakage), Vật thể nhỏ (Small Objects), và Khoảng cách Miền (Domain Gap).
*   **GIAI ĐOẠN 2: MÔ HÌNH (Architecture & Training):** Thiết kế và huấn luyện một kiến trúc mạng nơ-ron "Siamese-YOLOv8" tùy chỉnh, được tối ưu hóa cho tốc độ và khả năng so khớp mêtric, sử dụng hàm mất mát đa nhiệm (multi-task loss).
*   **GIAI ĐOẠN 3: HẬU KỲ (Post-Processing for STIoU):** Triển khai một pipeline suy luận thời gian thực, trong đó các phát hiện "thô" (raw) từ mô hình được đưa vào một bộ theo dõi (tracker) dựa trên Kalman Filter để làm mượt và nội suy, tối ưu hóa trực tiếp cho STIoU.
*   **GIAI ĐOẠN 4: TRIỂN KHAI (Acceleration for Jetson):** Biên dịch và tối ưu hóa toàn bộ pipeline (từ Giai đoạn 3) bằng cách sử dụng NVIDIA TensorRT và DeepStream SDK để đạt được hiệu suất > 15 FPS trên phần cứng Jetson.

## PHẦN 4: GIAI ĐOẠN 1 – PIPELINE DỮ LIỆU: TIỀN XỬ LÝ VÀ TĂNG CƯỜNG

Giai đoạn này đặt nền móng. Bằng cách đầu tư vào một pipeline dữ liệu phức tạp, chúng ta giảm tải gánh nặng tính toán cho mô hình tại thời điểm suy luận, cho phép một mô hình nhẹ hơn đạt được độ chính xác cao.

### 4.1 Chiến lược Phân chia Train/Validation Chống Rò rỉ Dữ liệu

Vấn đề rò rỉ dữ liệu (data leakage) là một nguy cơ có thật, như người dùng đã xác định. Việc chia ngẫu nhiên các khung hình (frames) sẽ khiến mô hình "nhìn thấy" các khung hình gần như giống hệt nhau trong cả tập huấn luyện và tập kiểm tra, dẫn đến điểm số validation cao giả tạo nhưng hiệu suất trên private test lại kém.

*   **Giải pháp:** Group-Based Splitting (Chia theo nhóm).
*   **Triển khai:** Phân tích tệp annotations.json và trích xuất một danh sách tất cả các video_id duy nhất (ví dụ: "Backpack_0", "Backpack_1", "Jacket_0", v.v.). Dựa trên dữ liệu mẫu, có 14 loại đối tượng huấn luyện, mỗi loại có 2 video, tổng cộng là 28 video. Thực hiện phép chia train/validation (ví dụ: 80/20 hoặc 70/30) trên danh sách 28 video_id này. Ví dụ: 22 video (và tất cả các khung hình liên quan của chúng) được đưa vào tập train, và 6 video còn lại (và tất cả các khung hình của chúng) được đưa vào tập val.
*   **Kết quả:** Đảm bảo rằng tập validation bao gồm các video hoàn toàn chưa từng thấy (unseen) trong quá trình huấn luyện, mô phỏng chính xác kịch bản đánh giá của public và private test set.

### 4.2 Tăng cường Dữ liệu (Augmentation) để Giải quyết Vật thể nhỏ (Thay thế SAHI)

Như đã phân tích trong 2.1, vì chúng ta không thể sử dụng SAHI, chúng ta phải dạy mô hình phát hiện vật thể nhỏ ngay từ đầu.

#### 4.2.1 Phương pháp 1: Tiling / Cropping (Cắt xén)

*   **Chiến lược:** Khai thác "sức mạnh của Tiling" (The Power of Tiling). Thay vì luôn huấn luyện trên toàn bộ khung hình (ví dụ: 1920 x 1080), bộ tải dữ liệu (data loader) sẽ, với một xác suất nhất định, cắt (crop) ngẫu nhiên một cửa sổ nhỏ hơn (ví dụ: 640 x 640) từ ảnh gốc.
*   **Tác động:** Thao tác này hoạt động như một "zoom kỹ thuật số" hiệu quả. Các vật thể nhỏ, trước đây chỉ chiếm vài pixel, giờ đây chiếm một phần tương đối lớn hơn của đầu vào. Điều này buộc mô hình phải học các đặc trưng chi tiết của chúng, thay vì bỏ qua chúng như nhiễu (noise).

#### 4.2.2 Phương pháp 2: Copy-Paste Augmentation

*   **Chiến lược:** Đây là một kỹ thuật cực kỳ hiệu quả để tăng số lượng các vật thể hiếm hoặc nhỏ.
*   **Pipeline:**
    1.  Sử dụng các nhãn (annotations) để cắt (crop) các pixel của một vật thể ground-truth (ví dụ: "Backpack_0" tại khung 3483) từ khung hình của nó.
    2.  Áp dụng các phép biến đổi hình học và màu sắc (xoay, thay đổi kích thước, thay đổi độ sáng) cho vật thể đã cắt.
    3.  Chọn một khung hình ngẫu nhiên khác làm hậu cảnh (background).
    4.  "Dán" (paste) vật thể này vào một vị trí hợp lý trên khung hình hậu cảnh và cập nhật nhãn (label).
*   **Tác động:** Giải quyết sự mất cân bằng dữ liệu (nhiều khung hình không có vật thể) và tăng đáng kể số lượng vật thể nhỏ, buộc mô hình phải học cách phát hiện chúng trong các bối cảnh (contexts) đa dạng.

### 4.3 Tăng cường Dữ liệu để Thu hẹp Khoảng cách Miền Ground-to-Aerial

Phần này của pipeline dữ liệu nhằm mục đích làm cho dữ liệu video fly cam trở nên "khó" và "biến đổi" hơn, chuẩn bị cho mô hình học các đặc trưng bất biến (invariant features) cần thiết để khớp với ảnh tham chiếu.

#### 4.3.1 Tăng cường Hình học (Geometric Augmentations)

*   **Chiến lược:** Dữ liệu drone về cơ bản là không có hướng "lên" cố định. Một fly cam bay từ phía Bắc hay phía Nam đều nhìn thấy cùng một vật thể, chỉ là bị xoay đi.
*   **Các phép biến đổi bắt buộc:**
    *   **Xoay (Rotation):** Áp dụng các phép xoay 90, 180, và 270 độ.
    *   **Lật (Flips):** Áp dụng lật ngang (Horizontal Flip) và lật dọc (Vertical Flip).
*   **Tác động:** Đây là các phép biến đổi thực tế (realistic transformations) cho dữ liệu trên không, tăng gấp 4 lần (hoặc hơn) tập dữ liệu huấn luyện một cách hiệu quả mà không tốn chi phí.

#### 4.3.2 Tăng cường Quang học & Màu sắc (Photometric Augmentations)

*   **Chiến lược:** Mô phỏng trực tiếp "điều kiện ánh sáng khắc nghiệt" được mô tả.
*   **Các phép biến đổi bắt buộc:**
    *   **Brightness/Contrast Jitter:** Thay đổi ngẫu nhiên độ sáng và độ tương phản.
    *   **Gamma Correction:** Thay đổi gamma để mô phỏng điều kiện nắng gắt (gamma < 1) hoặc thiếu sáng (gamma > 1).
    *   **Noise & Blur:** Thêm nhiễu Gaussian (Gaussian Noise) và mờ chuyển động (Motion Blur) để mô phỏng nhiễu cảm biến và chuyển động nhanh của drone.

#### 4.3.3 Tăng cường mô phỏng (Synthetic Augmentation) (Nâng cao)

*   **Chiến lược:** Nếu các phương pháp trên không đủ để bắc cầu qua khoảng cách miền, một chiến lược nâng cao là tạo ra dữ liệu tổng hợp (synthetic data).
*   **Triển khai:** Sử dụng các công cụ 3D (như Blender) hoặc các trình giả lập (như Unreal Engine / AirSim) để render (kết xuất) các mô hình 3D của các vật thể (ví dụ: ba lô, áo phao) từ góc nhìn từ trên xuống (top-down view).
*   **Tác động:** Điều này trực tiếp tạo ra dữ liệu huấn luyện "aerial-view" hoàn hảo cho các vật thể mà chúng ta chỉ có ảnh tham chiếu "ground-view", cung cấp các cặp (pair) lý tưởng cho việc học mêtric (metric learning).

## PHẦN 5: GIAI ĐOẠN 2 – KIẾN TRÚC MÔ HÌNH: "SIAMESE-YOLOV8"

Kiến trúc cốt lõi phải được thiết kế để cân bằng giữa tốc độ (Ràng buộc 1) và khả năng so khớp (Ràng buộc 3).

### 5.1 Lựa chọn Backbone: Tối ưu hóa cho Jetson

*   **Phân tích:** Chúng ta cần một backbone đã được chứng minh là có sự cân bằng tốt nhất giữa tốc độ trên thiết bị biên và chất lượng đặc trưng.
*   **Các ứng viên:**
    *   **ResNet-10/18:** Rất nhanh, nhưng các đặc trưng có thể quá nông (shallow) để phân biệt các vật thể tinh vi.
    *   **MobileNetV2/V3:** Lựa chọn tốt cho di động.
    *   **EfficientNet-B0:** Thường hiệu quả hơn MobileNet.
    *   **YOLOv8n Backbone (CSPDarknet):** Được cho là SOTA hiện tại về cân bằng tốc độ/độ chính xác trên các thiết bị biên.
*   **Quyết định:** Sử dụng backbone của YOLOv8n.
*   **Lợi thế chính:** Bằng cách chọn backbone của YOLOv8, chúng ta có thể khởi tạo mô hình của mình với các trọng số (weights) đã được huấn luyện trước trên COCO. Điều này cung cấp một bộ trích xuất đặc trưng mạnh mẽ ngay từ đầu, đã học được các khái niệm cơ bản về vật thể, kết cấu và hình dạng.

### 5.2 Xử lý Hình ảnh Tham chiếu: Tạo Query Vector

Chúng ta có 3 ảnh tham chiếu cho mỗi vật thể. Chúng cần được nén thành một vector đặc trưng (query vector) duy nhất.

#### 5.2.1 Feature Extractor (Bộ trích xuất Đặc trưng)

*   **Kiến trúc:** Chúng ta sử dụng cùng một backbone YOLOv8n (từ 5.1) để trích xuất đặc trưng từ 3 ảnh tham chiếu.
*   **Weight Sharing (Chia sẻ Trọng số):** Backbone xử lý ảnh tham chiếu và backbone xử lý video chia sẻ cùng một bộ trọng số. Đây là kiến trúc Siamese Network (mạng Xiêm) cổ điển.
*   **Tác động:** (1) Đảm bảo rằng các đặc trưng từ cả hai miền (ground và aerial) được chiếu vào cùng một không gian tiềm ẩn (latent space). (2) Giảm đáng kể số lượng tham số của mô hình.

#### 5.2.2 Aggregation Strategy (Chiến lược Gộp)

*   **Vấn đề:** Làm thế nào để kết hợp 3 vector đặc trưng (từ 3 ảnh) thành 1 vector?
*   **Lựa chọn 1 (Đơn giản):** Average Pooling. Tính trung bình 3 vector. Nhanh, nhưng có thể làm mờ các đặc trưng nếu một trong các ảnh tham chiếu bị mờ hoặc có góc chụp xấu.
*   **Lựa chọn 2 (Đề xuất):** Attention Pooling. Chúng ta thêm một mô-đun Multi-Head Attention nhỏ.
*   **Rationale:** 3 ảnh tham chiếu có thể có chất lượng và góc nhìn khác nhau. Attention pooling cho phép mô hình học cách gán trọng số (weights) cao hơn cho các đặc trưng từ ảnh "tốt nhất" (ví dụ: ảnh rõ nét, góc chụp tốt) và gán trọng số thấp (bỏ qua) các đặc trưng nhiễu từ các ảnh mờ. Điều này tạo ra một query vector tổng hợp mạnh mẽ và đáng tin cậy hơn.

### 5.3 Kiến trúc Đầu phát hiện (Head) Tùy chỉnh: Tích hợp Query vào YOLO

Đây là phần sửa đổi quan trọng nhất. Chúng ta loại bỏ đầu phát hiện (detection head) tiêu chuẩn của YOLOv8, vốn dự đoán 80 lớp COCO, và thay thế nó bằng một đầu phát hiện so khớp tương đồng (similarity matching head).

*   **Kiến trúc Đề xuất (Lấy cảm hứng từ):**
    1.  **Backbone (Video):** Khung hình video đi qua backbone YOLOv8n (đã chia sẻ trọng số) để tạo ra các feature map đa tỷ lệ (ví dụ: P3, P4, P5).
    2.  **Query (Tham chiếu):** 3 ảnh tham chiếu đi qua cùng backbone, được gộp (pool) bằng Attention Pooling (từ 5.2.2) để tạo ra một Query Vector $Q_v$ duy nhất.
    3.  **So khớp (Matching):** Tại mỗi lớp đặc trưng (P3, P4, P5), chúng ta tính toán Cosine Similarity (độ tương đồng cosin) giữa vector $Q_v$ và mọi vector đặc trưng (tại mỗi vị trí không gian x, y) trên feature map của video.
    4.  **Heatmap:** Kết quả của bước 3 là một "bản đồ nhiệt" (heatmap) về sự tương đồng cho mỗi tỷ lệ.
    5.  **Đầu phát hiện (Head) Tùy chỉnh:** Đầu phát hiện mới này (một vài lớp CNN nhỏ) bây giờ sẽ dự đoán 3 đầu ra tại mỗi vị trí:
        *   `objectness_score` (1-dim): Khả năng vị trí này chứa bất kỳ vật thể nào (lấy từ YOLO).
        *   `similarity_score` (1-dim): Điểm tương đồng từ heatmap (vật thể này có khớp với $Q_v$ không?).
        *   `bbox_regression` (4-dim): 4 giá trị $(x, y, w, h)$ để tinh chỉnh bounding box.
*   **Tác động:** Kiến trúc này biến YOLO từ một bộ phân loại (classifier) nhiều lớp thành một bộ so khớp (matcher) nhị phân. Nó không còn hỏi "Đây là lớp gì?" mà hỏi "Vật thể ở đây có giống vật thể tham chiếu không?". Đây chính xác là bản chất của bài toán Zero-Shot / Reference-based Detection.

## PHẦN 6: GIAI ĐOẠN 3 – CHIẾN LƯỢC HUẤN LUYỆN ĐA NHIỆM (MULTI-TASK)

Để huấn luyện mô hình (end-to-end), chúng ta cần một hàm mất mát kết hợp (combined loss function) để dạy cho cả hai nhiệm vụ: phát hiện (detection) và so khớp (similarity).

### 6.1 Hàm mất mát (Loss Function) Kết hợp

Hàm mất mát tổng thể sẽ là tổng có trọng số của hai thành phần:

$L_{total} = w_1 \cdot L_{detection} + w_2 \cdot L_{similarity}$

trong đó $w_1$ và $w_2$ là các siêu tham số (hyperparameters) để cân bằng hai nhiệm vụ.

### 6.2 Loss 1: Detection Loss (Mất mát Phát hiện)

Phần này được lấy trực tiếp từ các hàm mất mát tiêu chuẩn của YOLOv8. Nó được áp dụng cho các dự đoán có `similarity_score` cao.

*   **Box Loss:** Sử dụng một hàm mất mát bounding box hiện đại như CIoU (Complete IoU) hoặc SIoU để tối ưu hóa vị trí $(x, y, w, h)$.
*   **Objectness Loss:** Sử dụng Focal Loss hoặc Binary Cross-Entropy (BCE) để huấn luyện `objectness_score` (phân biệt vật thể với hậu cảnh).

### 6.3 Loss 2: Similarity Loss (Mất mát Tương đồng)

Đây là thành phần quan trọng nhất để giải quyết Khoảng cách Miền (Ràng buộc 3).

*   **Lựa chọn:** Triplet Loss (Mất mát Bộ ba).
*   **So sánh Triplet vs. Contrastive:** Mặc dù Contrastive Loss (dùng cho các cặp) là một lựa chọn, Triplet Loss (từ FaceNet) thường vượt trội hơn cho các tác vụ retrieval (truy xuất) và nhận dạng (recognition). Triplet Loss tối ưu hóa thứ hạng tương đối (relative ranking) và tập trung vào các "hard negatives" (mẫu âm khó), điều này rất quan trọng để phân biệt, ví dụ, "Backpack_0" (mục tiêu) với "Backpack_1" (vật thể khác cùng loại).
*   **Triển khai "Triplet Mining" (Khai thác Bộ ba):**
    *   **Anchor (A) (Mỏ neo):** Vector đặc trưng của ground-truth bounding box (vật thể mục tiêu) trong một khung hình video (ví dụ: crop của "Backpack_0" từ video fly cam).
    *   **Positive (P) (Dương tính):** Query vector $Q_v$ đã được tính toán của vật thể đó (ví dụ: embedding gộp từ 3 ảnh tham chiếu mặt đất của "Backpack_0").
    *   **Negative (N) (Âm tính):** Vector đặc trưng của một "hard negative" – tức là một vật thể khác trong cùng khung hình (ví dụ: crop của "Jacket_1" hoặc "Backpack_1").
*   **Mục tiêu Loss:** Huấn luyện mô hình để đảm bảo:
    $Distance(A, P) + margin < Distance(A, N)$
    Điều này buộc mô hình phải học các đặc trưng sao cho khoảng cách giữa biểu diễn aerial (A) và biểu diễn ground (P) của cùng một vật thể nhỏ hơn khoảng cách giữa biểu diễn aerial (A) và biểu diễn của một vật thể khác (N), cộng thêm một lề (margin).

## PHẦN 7: GIAI ĐOẠN 4 – PIPELINE SUY LUẬN: TỐI ƯU HÓA STIOU & JETSON

Đây là pipeline "tuyệt đối" được thực thi trên NVIDIA Jetson tại thời điểm dự đoán (runtime).

### 7.1 Bước 1: Khởi tạo (Initialization) (Thực hiện một lần cho mỗi video)

1.  Tải (load) 3 ảnh tham chiếu cho vật thể mục tiêu (ví dụ: img_1.jpg, img_2.jpg, img_3.jpg của "Backpack_0").
2.  Đưa 3 ảnh này qua một lần qua backbone của mô hình Siamese-YOLOv8 (đã được tối ưu hóa bằng TensorRT).
3.  Áp dụng lớp Attention Pooling (từ 5.2.2) để tạo ra một Query Vector $Q_v$ duy nhất, tĩnh (static).
4.  Khởi tạo một instance của SORT Tracker. Tracker này chứa một danh sách rỗng các "tracks" (đối tượng đang theo dõi) và một Kalman Filter được cấu hình cho mỗi track mới.

### 7.2 Bước 2: Vòng lặp Suy luận trên từng Khung hình (Per-Frame Loop) (Thực hiện lặp lại cho mỗi khung hình f của video)

1.  **Phát hiện Thô (Raw Detection):** Đưa khung hình f và query vector $Q_v$ (đã tạo ở 7.1) vào mô hình Siamese-YOLOv8. Mô hình trả về một danh sách các phát hiện thô (raw detections) $D_f = [(box_1, sim_1), (box_2, sim_2),...]$ đã được lọc qua một ngưỡng tin cậy (ví dụ: `similarity_score` > 0.5).
2.  **Xử lý Hậu kỳ (Post-Processing) Tối ưu STIoU:**
    a.  **Dự đoán (Prediction) (Kalman):** Yêu cầu SORT tracker dự đoán vị trí mới (t) của tất cả các tracks cũ (từ t-1) dựa trên trạng thái (vận tốc) đã học của chúng.
    b.  **Liên kết (Association) (Hungarian Algorithm):** So khớp (match) danh sách phát hiện thô $D_f$ (từ mô hình) với danh sách các tracks đã dự đoán (từ Kalman) bằng cách sử dụng IoU làm ma trận chi phí (cost matrix).
    c.  **Cập nhật (Update) (Kalman):**
        *   **Các Tracks đã khớp (Matched Tracks):** Vị trí phát hiện thô $D_f$ được đưa vào Kalman Filter của track tương ứng. Kalman Filter sẽ cập nhật trạng thái của nó, cho ra một vị trí đã được làm mượt (smoothed).
        *   **Các Tracks không khớp (Unmatched Tracks) (Vật thể bị che khuất):** Không có phát hiện nào khớp. Tăng bộ đếm "age". Quan trọng nhất: tracker sẽ xuất ra vị trí đã được dự đoán (predicted) bởi Kalman Filter. Đây là bước "lấp đầy" (interpolation).
        *   **Các Phát hiện không khớp (Unmatched Detections) (Vật thể mới):** Một phát hiện $D_f$ mới không khớp với track nào. Tạo một track mới (với Kalman Filter mới) cho nó.
3.  **Đầu ra Cuối cùng:** Chỉ xuất ra các bounding box từ các tracks đang hoạt động (active tracks) (ví dụ: các tracks đã được xác nhận trong k khung hình và "age" < `max_age`).

#### 7.2.4 Bảng: Tác động của Xử lý Hậu kỳ (Post-Processing) đến STIoU

Bảng sau đây minh họa tầm quan trọng của Bước 7.2.3 trong việc tối đa hóa điểm số STIoU.

| Khung hình (Frame) | Phát hiện Thô (Raw Detection) (Từ YOLO) | Trạng thái SORT/Kalman | Đầu ra Cuối cùng (Final Output) | Tác động STIoU |
| :--- | :--- | :--- | :--- | :--- |
| f | Box (x, y, w, h) (Hơi giật) | Track 1 (Matched) | Box (x', y', w', h') (Đã làm mượt) | Tăng Tử số (IoU cao hơn do ổn định) |
| f+1 | Không có (Missed Detection) | Track 1 (Unmatched, Predicted) | Box ($x_{pred}$, $y_{pred}$,...) (Đã nội suy) | Ngăn Mẫu số tăng (lấp đầy lỗ hổng) |
| f+2 | Box ($x_2, y_2, w_2, h_2$) (Xuất hiện lại) | Track 1 (Matched, Re-acquired) | Box (x'', y'', w'', h'') (Đã làm mượt) | Tăng Tử số (IoU cao hơn) |
| f+3 | Box ($x_{noise}, y_{noise}$,...) (False Positive) | Unmatched Detection (Mới, "age"=0) | Không có (Bị lọc do track chưa_xác nhận) | Ngăn Mẫu số tăng (lọc nhiễu) |

### 7.3 Bước 3: Tối ưu hóa Triển khai trên Jetson (Acceleration)

Một script Python chạy PyTorch và OpenCV (dùng `cv2.VideoCapture()`) sẽ không bao giờ đạt >15 FPS do các điểm nghẽn (bottlenecks) ở CPU, giải mã video và sao chép bộ nhớ.

#### 7.3.1 Tối ưu hóa Mô hình: Pytorch -> ONNX -> TensorRT

1.  **Huấn luyện:** Mô hình Siamese-YOLOv8 được huấn luyện bằng PyTorch.
2.  **Xuất (Export):** Mô hình được xuất sang định dạng ONNX (Open Neural Network Exchange).
3.  **Biên dịch (Compile):** Sử dụng công cụ `trtexec` của NVIDIA để phân tích cú pháp (parse) tệp ONNX và biên dịch nó thành một "engine" TensorRT (.engine).
4.  **Lượng tử hóa (Quantization):** Bước này là bắt buộc. Khi biên dịch, phải sử dụng cờ (flag) `--fp16`. Thao tác này chuyển đổi các trọng số 32-bit (FP32) sang 16-bit (FP16), cho phép mô hình chạy trên các Lõi Tensor (Tensor Cores) chuyên dụng của Jetson, thường dẫn đến tăng tốc độ 2-3 lần.

#### 7.3.2 Xử lý các Layer Tùy chỉnh (Custom Layers) (Phần khó nhất)

*   **Vấn đề:** Quá trình `export()` sang ONNX/TensorRT tiêu chuẩn sẽ thất bại. Lý do là mô hình của chúng ta chứa các hoạt động (operations) tùy chỉnh (ví dụ: Attention Pooling từ 5.2.2, Cosine Similarity Head từ 5.3) mà ONNX và TensorRT không hiểu.
*   **Giải pháp:** Chúng ta phải viết một TensorRT Plugin (sử dụng giao diện IPluginV3 C++).
*   **Triển khai:** Đây là một thư viện C++ tùy chỉnh (.so file) mà chúng ta tự viết, triển khai logic của các layer tùy chỉnh của mình bằng CUDA. Khi trình phân tích cú pháp (parser) TensorRT gặp một nút (node) không xác định (ví dụ: "CustomAttentionPooling"), chúng ta "đăng ký" (register) plugin của mình để cung cấp việc triển khai đã được tối ưu hóa bằng GPU cho nút đó.

#### 7.3.3 Tối ưu hóa Pipeline: NVIDIA DeepStream

*   **Vấn đề:** Ngay cả với một TensorRT engine nhanh, script Python vẫn là điểm nghẽn.
*   **Giải pháp:** Triển khai toàn bộ pipeline (từ 7.1 đến 7.2) bên trong NVIDIA DeepStream SDK. DeepStream là một framework dựa trên GStreamer được thiết kế để xử lý video AI hiệu suất cao, giảm thiểu việc sử dụng CPU.
*   **Kiến trúc DeepStream:** Pipeline sẽ trông như sau:
    `Gst-filesrc` (Đọc tệp video) -> `Gst-decodebin` (Sử dụng bộ giải mã phần cứng NVDEC) -> `Gst-nvstreammux` (Tạo batch các khung hình) -> `Gst-nvinfer` (Chạy engine TensorRT tùy chỉnh của chúng ta (với plugin tùy chỉnh từ 7.3.2)) -> `Gst-nvtracker` (Thực hiện theo dõi đối tượng) -> `Gst-fakesink` (Xuất kết quả JSON)
*   **Tích hợp then chốt:** Bước xử lý hậu kỳ (7.2.3) được thực hiện bởi `Gst-nvtracker`. Chúng ta có thể cấu hình plugin này để sử dụng một thư viện theo dõi cấp thấp. Bằng cách cung cấp một triển khai (implementation) C++ của SORT/Kalman filter và liên kết nó với `Gst-nvtracker`, chúng ta thực hiện toàn bộ vòng lặp (detect + track) trên GPU và các bộ tăng tốc phần cứng, loại bỏ hoàn toàn Python khỏi đường dẫn nóng (hot-path) và đảm bảo đạt được mục tiêu > 15 FPS.

## PHẦN 8: KẾT LUẬN VÀ LỘ TRÌNH TRIỂN KHAI

### 8.1 Tóm tắt "Pipeline Tuyệt đối"

Chiến lược end-to-end được đề xuất để chiến thắng AeroEyes 2025 là một cách tiếp cận đa tầng, tích hợp:

*   **Dữ liệu:**
    *   **Phân chia (Split):** tập train/val theo `video_id` để ngăn rò rỉ.
    *   **Tăng cường (Augment):** mạnh mẽ bằng Tiling, Copy-Paste, và xoay 90/180/270 độ để giải quyết vấn đề vật thể nhỏ và mô phỏng góc nhìn drone.
*   **Mô hình:**
    *   Xây dựng một Siamese-YOLOv8n với backbone chia sẻ trọng số.
    *   **Xử lý Tham chiếu:** Tạo Query Vector từ 3 ảnh tham chiếu bằng cách sử dụng backbone và một lớp Attention Pooling để gộp các đặc trưng.
    *   **Huấn luyện:** Sử dụng hàm mất mát đa nhiệm (multi-task loss) kết hợp: Detection Loss (CIoU + Focal) để tìm vật thể và Triplet Loss để học sự tương đồng (thu hẹp khoảng cách miền ground-to-aerial).
*   **Suy luận:**
    1.  Tạo Query Vector một lần.
    2.  Chạy mô hình Siamese-YOLOv8 trên từng khung hình để lấy phát hiện thô.
    3.  Đưa phát hiện thô vào SORT Tracker (với Kalman Filter).
*   **Tối ưu hóa STIoU:** Đầu ra cuối cùng là từ SORT tracker, vốn đã làm mượt (smoothed) các bounding box, nội suy (interpolated) các khung hình bị thiếu, và lọc (filtered) các phát hiện sai.
*   **Triển khai Jetson:** Biên dịch mô hình sang TensorRT FP16 (yêu cầu Plugin C++ tùy chỉnh cho các layer không tiêu chuẩn) và triển khai toàn bộ pipeline trong DeepStream.

### 8.2 Rủi ro và Giảm thiểu

*   **Rủi ro:** Viết Plugin C++ cho TensorRT (7.3.2) rất phức tạp và tốn thời gian.
    *   **Giảm thiểu:** Bắt đầu bằng cách triển khai toàn bộ pipeline trong Python + PyTorch. Đo lường FPS. Nếu chậm, bước 1 là biên dịch mô hình sang TensorRT (sử dụng các ops tiêu chuẩn nếu có thể). Chỉ khi FPS vẫn < 15, mới đầu tư thời gian vào việc viết plugin C++ tùy chỉnh.
*   **Rủi ro:** Triplet Loss (6.3) nổi tiếng là khó hội tụ.
    *   **Giảm thiểu:** Bắt đầu bằng các trọng số đã được huấn luyện trước trên COCO. Sử dụng các kỹ thuật "hard negative mining" (khai thác mẫu âm khó) và tinh chỉnh cẩn thận siêu tham số margin (lề).

### 8.3 Lời kết

Chiến lược này vượt trội hơn các phương pháp tiếp cận ngây thơ (naive) vì nó được thiết kế toàn diện để giải quyết tất cả các ràng buộc của bài toán. Nó không chỉ đơn thuần là chọn "mô hình tốt nhất". Nó sử dụng tăng cường dữ liệu để bù đắp cho các ràng buộc về hiệu suất (loại bỏ SAHI), sử dụng hàm mất mát (Triplet Loss) để giải quyết các ràng buộc về dữ liệu (khoảng cách miền), và sử dụng xử lý hậu kỳ (Kalman Filter) để tối ưu hóa trực tiếp cho chỉ số đánh giá (STIoU). Cách tiếp cận này cân bằng một cách có hệ thống giữa độ chính xác và hiệu suất thời gian thực, cung cấp một pipeline "tuyệt đối" để triển khai trên NVIDIA Jetson.