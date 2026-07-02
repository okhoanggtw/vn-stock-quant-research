# Sử dụng Python 3.10 slim làm base image để giảm dung lượng
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (gcc, curl để lấy pip...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container
COPY . .

# Mặc định khi chạy container sẽ thực thi pipeline nghiên cứu định lượng
CMD ["python", "scripts/07_pipeline.py"]
