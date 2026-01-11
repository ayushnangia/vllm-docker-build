# Custom Dockerfile for sglang commit 6f560c76 (PR #117 - January 30, 2024)
# "Improve the control of streaming and improve the first token latency in streaming"
#
# Requirements from that era:
# - vllm >= 0.2.5 (which requires PyTorch 2.1.1, CUDA 11.8)
# - Python 3.10
# - torch 2.1.1 to match vllm

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.1 for CUDA 11.8
RUN pip3 install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# Install vllm 0.2.5 (from that era)
RUN pip3 install vllm==0.2.5

# Set workspace
WORKDIR /sgl-workspace

# Download sglang at specific commit via GitHub tarball API
# (git fetch doesn't work for this old commit)
ARG SGLANG_COMMIT=6f560c761b2fc2f577682d0cfda62630f37a3bb0
RUN curl -L "https://github.com/sgl-project/sglang/archive/${SGLANG_COMMIT}.tar.gz" -o sglang.tar.gz && \
    tar -xzf sglang.tar.gz && \
    mv sglang-${SGLANG_COMMIT} sglang && \
    rm sglang.tar.gz

# Install sglang from source
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Write commit proof
RUN echo "${SGLANG_COMMIT}" > /opt/sglang_commit.txt

# Verify installation
RUN python3 -c "import sglang; print('SGLang imported successfully')"

WORKDIR /sgl-workspace/sglang
