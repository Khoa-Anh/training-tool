# CV TrainingTool

## 1. Sơ đồ

![Untitled](images/Untitled.png)

- **Giao diện trên eSales sẽ tích hợp nút training để người dùng tự huấn luyện, có các tham số mặc định kèm với tùy chỉnh cơ bản.**
- **Việc huấn luyện được thực hiện trên cloud VMs, thời gian vài tiếng, có in ra các log để kiểm tra quá trình huấn luyện.**
- **Người dùng sẽ tự kiểm thử và đánh giá.**

## 2. Cấu hình tham khảo cho dịch vụ **Cloud GPU**

- **NVIDIA T4 Tensor Core GPU for AI Inference**
    
    [NVIDIA T4 Tensor Core GPUs for Accelerating AI Inference](https://www.nvidia.com/en-us/data-center/tesla-t4/)
    
- **NVIDIA V100 Tensor Core**
    
    [NVIDIA V100 | NVIDIA](https://www.nvidia.com/en-us/data-center/v100/)
    
- **NVIDIA A100 Tensor Core GPU**
    
    [A100 GPU's Offer Power, Performance, & Efficient Scalability](https://www.nvidia.com/en-us/data-center/a100/)
    
- **Cấu hình tối thiểu: NVIDIA T4 Tensor Core GPU for AI Inference**
    - **GPU Memory: 16 GB GDDR6**
- **Cấu hình trung bình: NVIDIA V100 Tensor Core**
    - **GPU Memory: 32GB CoWoS HBM2**
- **Cấu hình trung bình: NVIDIA A100 Tensor Core GPU**
    - **GPU Memory: 40GB CoWoS HBM2**
- Nên sử dụng từ NVIDIA V100 Tensor Core hoặc tương đương để xử lý tốt chức năng huấn luyện

## **3 Hướng dẫn chi tiết sử dụng ứng dụng Streamlit cho việc huấn luyện và phát hiện đối tượng**

- Ứng dụng này được thiết kế để hỗ trợ nhập dữ liệu, huấn luyện mô hình và phát hiện đối tượng.
- Có 3 bước:
    - Nhập dữ liệu
    - Huấn luyện
    - Phát hiện đối tượng

### Bước 1: Nhập Dữ liệu

1. **Chọn phương thức nhập**
- Mở ứng dụng và chọn "Import Database" từ thanh điều hướng bên trái.
- Chọn phương thức nhập dữ liệu: "Import from Custom Vision" hoặc "Import from a ZIP file".
1. **Nhập từ Custom Vision**
- Chọn CustomVisionResource [S0]
- Chọn dự án cần huấn luyện > Nhấn nút **Project Settings** bên góc phải trên cùng của màn hình
- Lưu project ID và training key của dự án:
- Demo:
    - Project Name: JJVN_TBAI
    - project ID: 58af6408-2737-4027-8e1a-fde72efd9ebf
    - Key: 14f8cc99ea3648a6a2ccc49c5946e419
        
        ![Untitled](images/Untitled%201.png)
        
    
    ![Untitled](images/Untitled%202.png)
    
- Tại ứng dụng TrainingTool, nhập project ID và training key.
- Đặt tỷ lệ chia tập huấn luyện và kiểm thử trong trường "Enter validation split ratio".
- Nhấn "Download" để tải dữ liệu. Hệ thống sẽ xóa dữ liệu cũ, tạo thư mục mới và tải dữ liệu về.
- Dữ liệu được chia thành tập huấn luyện và kiểm thử theo tỷ lệ đã nhập.

![Untitled](images/Untitled%203.png)

- Đặt tỷ lệ chia tập huấn luyện và kiểm thử trong trường "Enter validation split ratio".
- Nhấn "Download" để tải dữ liệu. Hệ thống sẽ xóa dữ liệu cũ, tạo thư mục mới và tải dữ liệu về.
- Dữ liệu được chia thành tập huấn luyện và kiểm thử theo tỷ lệ đã nhập.

![Untitled](images/Untitled%204.png)

1. **Nhập từ tệp ZIP**
- Tải lên tệp ZIP chứa hình ảnh và nhãn.
- Tệp ZIP sẽ được giải nén tự động vào thư mục "data".
- Hệ thống sẽ tạo bản đồ lớp từ "data/data.yaml" sang "data/mapping.txt".
- Thông báo thành công sẽ hiển thị sau khi tải và giải nén thành công.

![Untitled](images/Untitled%205.png)

### Bước 2: Huấn luyện

1. **Cài đặt tham số huấn luyện**
- Chọn "Training" từ thanh điều hướng.
- Điền các thông số như số epoch, kích thước batch, tỷ lệ học, v.v.
- Các thông số này bao gồm:
- Epochs: Số lần lặp qua toàn bộ dữ liệu.
- Patience: Số epoch chờ đợi không cải thiện trước khi dừng sớm.
- Batch Size: Kích thước lô dữ liệu trong một lần cập nhật tham số.
- Image Size: Kích thước hình ảnh đầu vào.
- Optimizer: Phương pháp tối ưu hóa.
- Initial Learning Rate: Tốc độ học ban đầu.
- Final Learning rate: Tốc độ học cuối cùng.
1. **Bắt đầu huấn luyện**
- Nhấn "Start Training" để bắt đầu quá trình huấn luyện.

![Untitled](images/Untitled%206.png)

- Mô hình tốt nhất sẽ được lưu lại sau khi huấn luyện.

![Untitled](images/Untitled%207.png)

### Bước 3: Phát hiện Đối tượng

1. **Cấu hình và tải mô hình**
- Chọn "Object Detection" từ thanh điều hướng.
- Tải lên hình ảnh và chọn mô hình đã huấn luyện (.pt file).
- Điều chỉnh ngưỡng tin cậy của mô hình.
1. **Phát hiện đối tượng**
- Nhấn "Detect Objects" để bắt đầu phát hiện.
- Hình ảnh sẽ được hiển thị với các đối tượng được đánh dấu.
- Thông tin chi tiết về các đối tượng được phát hiện sẽ được hiển thị, bao gồm tên lớp và số lượng.
    
    ![Untitled](images/Untitled%208.png)
    

![Untitled](images/Untitled%209.png)

**Lưu ý khi sử dụng:**

- Đảm bảo rằng các thư mục dữ liệu cần thiết đã được tạo và cấu hình đúng trước khi bắt đầu các bước nhập dữ liệu hoặc huấn luyện.
- Kiểm tra và xử lý các lỗi có thể xảy ra trong quá trình nhập dữ liệu và huấn luyện để đảm bảo quá trình diễn ra suôn sẻ.
- Các thông số như tỷ lệ học, số epoch, và kích thước batch có thể cần được điều chỉnh tùy theo đặc thù của dữ liệu và yêu cầu của mô hình.

Ứng dụng này hỗ trợ người dùng từ bước nhập dữ liệu ban đầu cho đến huấn luyện và phát hiện đối tượng, qua đó giúp đơn giản hóa quá trình làm việc cho các nhiệm vụ phát hiện đối tượng.