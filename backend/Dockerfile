FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY main.py .

# Copy model weights
COPY model/best.pt /app/model/best.pt

# Set environment variables
ENV MODEL_PATH=/app/model/best.pt

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
