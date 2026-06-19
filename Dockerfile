# ============================================================
# Dockerfile — 数字图像处理 Flask Web 应用
# ============================================================
# 构建镜像:
#   docker build -t mosaic-vision .
#
# 运行容器:
#   docker run -d -p 5000:5000 --name mosaic-vision mosaic-vision
#
# 或使用 docker-compose:
#   docker compose up -d
# ============================================================

# ---- 基础镜像 ----
FROM python:3.11-slim

# ---- 设置环境变量 ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=web_app.py \
    FLASK_ENV=production

# ---- 安装系统依赖 ----
# ffmpeg: 视频格式转换（mp4v → H.264）
# libgl1-mesa-glx libglib2.0-0: OpenCV 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# ---- 创建工作目录 ----
WORKDIR /app

# ---- 安装 Python 依赖 ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 复制项目文件 ----
COPY . .

# ---- 创建输出目录 ----
RUN mkdir -p output/web_generated output/web_uploads output/web_captcha

# ---- 暴露端口 ----
EXPOSE 5000

# ---- 启动 Flask 应用 ----
# 生产环境建议使用 gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "web_app:app"]
