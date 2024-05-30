# Use a specific PyTorch base image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        make \
        gcc \
        libc-dev \
        musl-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        tesseract-ocr \
        libtesseract-dev \
        libreoffice-common \
        wkhtmltopdf \
        libreoffice-writer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Git and build-essential
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set noninteractive mode
ARG DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /trainer

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Set the entry point
ENTRYPOINT ["python", "-m", "task"]
