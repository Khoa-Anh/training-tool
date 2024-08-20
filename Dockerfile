# Sử dụng image chứa Python 3.12.2
FROM python:3.12.2


# Thêm các gói cần thiết cho OpenGL
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy chỉ requirements.txt để cài đặt dependencies trước
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn ứng dụng
COPY . .

EXPOSE 80

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
