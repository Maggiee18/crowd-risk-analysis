# ============================================================
# Stage 1: Build React frontend
# ============================================================
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --legacy-peer-deps
COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python backend + serve frontend
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# System deps (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY config.json .
COPY main.py .

# Copy YOLOv8 model if present
COPY yolov8n.pt* ./

# Copy built frontend
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create output dirs
RUN mkdir -p models output data

# Expose ports
EXPOSE 8000

# Environment
ENV PYTHONUNBUFFERED=1
ENV ADMIN_TOKEN=crowd2024admin

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
