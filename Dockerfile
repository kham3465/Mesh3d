# --- Giai đoạn 1: Cài đặt các dependency ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho build C++/Cython và các thư viện Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt và cài đặt thư viện Python để tận dụng cache
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Sao chép các mã nguồn C++/Cython cần thiết cho việc biên dịch
COPY FaceBoxes/ FaceBoxes/
COPY Sim3DR/ Sim3DR/
COPY utils/asset/render.c utils/asset/render.c

# Biên dịch tất cả các thành phần C++/Cython trong một layer duy nhất
RUN cd FaceBoxes/utils && python build.py build_ext --inplace && \
    cd ../../Sim3DR && python setup.py build_ext --inplace && \
    cd .. && gcc -shared -Wall -O3 -fPIC utils/asset/render.c -o utils/asset/render.so

# --- Giai đoạn 2: Xây dựng image cuối cùng ---
FROM python:3.10-slim

# Khai báo các biến có thể được truyền vào lúc build
ARG APP_ENV=prod
ARG PORT=8080

# Đặt các biến môi trường cho Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=${APP_ENV}
ENV PORT=${PORT}

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho runtime (ví dụ: OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép các thư viện đã cài đặt từ giai đoạn 'builder'
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Sao chép các file thực thi (như uvicorn, gunicorn, etc.) đã được pip cài đặt
COPY --from=builder /usr/local/bin /usr/local/bin

# Sao chép mã nguồn và tài nguyên cần thiết cho ứng dụng chạy
COPY api_server.py .
COPY db_utils.py .
COPY TDDFA_ONNX.py .
COPY bfm/ ./bfm/
COPY utils/ ./utils/
COPY models/ models/
COPY weights/ weights/
COPY configs/ configs/
COPY recognition/ ./recognition/
# Only copy the specific checkpoint needed for inference, not the whole results folder.
RUN mkdir -p results/checkpoints
COPY results/checkpoints/latest_recog_ck.onnx* results/checkpoints/

# Sao chép toàn bộ các thư mục mã nguồn đã được biên dịch từ giai đoạn builder
COPY --from=builder /app/FaceBoxes/ ./FaceBoxes/
COPY --from=builder /app/Sim3DR/Sim3DR_Cython*.so ./Sim3DR/
COPY --from=builder /app/utils/asset/render.so ./utils/asset/

# Expose cổng mà API server sẽ chạy, được định nghĩa bởi biến môi trường
EXPOSE ${PORT}

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
