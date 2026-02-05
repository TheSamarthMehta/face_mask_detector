# Dockerfile for Face Mask Detection Application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY task7_deployment/ ./task7_deployment/
COPY models/ ./models/
COPY task5_frontend/ ./task5_frontend/

# Expose ports
EXPOSE 8000
EXPOSE 5000

# Set environment variables
ENV MODEL_PATH=models/best_model.h5
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "task7_deployment/api.py"]
