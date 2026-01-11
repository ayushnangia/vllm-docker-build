# Dockerfile for SGLang commit bb3a3b6675b1844a13ebe368ad693f3dc75b315b
# Date: February 3, 2024
# Architecture: linux/amd64 (GPU)

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG SGLANG_COMMIT=bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch
RUN pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Copy ONLY the source for this specific commit
WORKDIR /sgl-workspace
COPY . /sgl-workspace/sglang

# Build flashinfer from source (old versions not on PyPI for Python 3.10)
RUN pip3 install ninja numpy && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && git checkout v0.1.2 && \
    cd python && TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MAX_JOBS=64 pip3 install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# Patch pyproject.toml to skip flashinfer install (already built)
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang package
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Write commit hash for verification (Properly using ARG)
RUN mkdir -p /opt && echo "$SGLANG_COMMIT" > /opt/sglang_commit.txt

WORKDIR /sgl-workspace
CMD ["python3", "-c", "import sglang; print('SGLang loaded successfully')"]
