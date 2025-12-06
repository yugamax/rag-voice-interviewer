# Python base image
# Python base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Non-interactive apt and minimal system deps. Conservative set: keep build tools
# and OpenBLAS so numerical libs (faiss, sentence-transformers) install reliably.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libopenblas-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Avoid pulling TensorFlow when using Transformers with PyTorch
ENV TRANSFORMERS_NO_TF=1
ENV PORT=8000

# Copy requirement file and install (upgrade pip first)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (must match your PORT env)
EXPOSE 7860

# Use HF-provided PORT, fallback to 7860 just in case
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]