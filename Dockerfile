# ==============================================================================
# Dockerfile for PeopleVision AI
# Multi-purpose image for running the FastAPI API and the Streamlit Dashboard
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set root application directory
WORKDIR /workspace

# Install system dependencies required by OpenCV, YOLO inference, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements first to utilize Docker build layer caching
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache YOLOv8n weights into the container image to prevent
# startup delays and dependency on external networks at runtime.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application package directories and project code
COPY app/ /workspace/app/
COPY dashboard/ /workspace/dashboard/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default runtime command. Can be overridden in docker-compose for dashboard.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
