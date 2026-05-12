# ============================================
# 阶段1: 构建阶段 - 下载依赖和模型
# ============================================
FROM python:3.11-slim AS builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖（用于编译Python包和下载模型）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    unzip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖（使用--no-cache-dir减少镜像大小）
RUN pip install --no-cache-dir -r requirements.txt

# 创建目录结构
RUN mkdir -p models photos /data

# 下载并解压人脸识别模型（buffalo_l）
RUN curl -fsSL -o /tmp/buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" && \
    unzip -oq /tmp/buffalo_l.zip -d models/ && \
    rm /tmp/buffalo_l.zip

# ============================================
# 阶段2: 运行阶段 - 只保留必要文件
# ============================================
FROM python:3.11-slim AS runner

# 设置工作目录
WORKDIR /app

# 安装运行时系统依赖（只保留必要的）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 从构建阶段复制模型文件
COPY --from=builder /app/models /app/models

# 创建必要的目录
RUN mkdir -p photos /data && chmod 777 /data photos

# 复制应用代码
COPY app/ /app/app/
COPY static/ /app/static/

# 设置环境变量（生产环境配置）
ENV PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/face_recognition.db \
    PHOTOS_DIR=/app/photos \
    PORT=8000

# 暴露端口（与应用监听端口一致）
EXPOSE 8000

# 健康检查（用于容器编排平台检测服务状态）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 生产环境启动命令（使用uvicorn，不带reload）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
