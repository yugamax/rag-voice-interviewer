# Python base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# System dependencies (if you hit errors with faiss or sentence-transformers, you can add more libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
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
EXPOSE ${PORT}

# Run the app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]