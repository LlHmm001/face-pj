FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

COPY . .

RUN mkdir -p models photos && \
    curl -L -o models/buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" && \
    unzip -o models/buffalo_l.zip -d models/ && \
    rm models/buffalo_l.zip && \
    mkdir -p /data && \
    chmod 777 /data photos

ENV DATABASE_URL=sqlite:////data/face_recognition.db
ENV PHOTOS_DIR=/app/photos

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
