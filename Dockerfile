# Use PyTorch image with CUDA support (already includes torch + CUDA runtime)
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Install system dependencies (ffmpeg is required by Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY app app

# Expose port 8000 for API
EXPOSE 8000

# Start FastAPI (with Uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
