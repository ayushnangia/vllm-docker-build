# SGLang Dockerfile for commit 22a6b9fc051154347b6eb5064d2f6ef9b4dba471
# Date: 2025-06-13
# Dependencies: torch==2.7.1, flashinfer_python==0.2.6.post1, sgl-kernel==0.1.7

# Base image for torch 2.7.x with CUDA 12.6
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    cmake ninja-build ccache \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.7.1 with CUDA 12.6
RUN pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install sgl-kernel from PyPI (version 0.1.7 is available)
RUN pip3 install sgl-kernel==0.1.7

# Build flashinfer from source (no wheel for torch 2.7)
RUN pip3 install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# Install xformers for torch 2.7 (optional but recommended)
RUN pip3 install xformers --index-url https://download.pytorch.org/whl/cu126 || echo "xformers installation failed, continuing without it"

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=22a6b9fc051154347b6eb5064d2f6ef9b4dba471

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 22a6b9fc051154347b6eb5064d2f6ef9b4dba471

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="22a6b9fc051154347b6eb5064d2f6ef9b4dba471" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang/python && \
    sed -i 's/"flashinfer[^"]*",*//g' pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' pyproject.toml && \
    sed -i 's/\[,/[/g' pyproject.toml && \
    sed -i 's/,\]/]/g' pyproject.toml

# First install torchao separately (required by runtime_common)
RUN pip3 install torchao==0.9.0 || pip3 install torchao

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Verify installation
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel OK')" && \
    python3 -c "import xformers; print('xformers OK')" 2>/dev/null || echo "xformers not available" && \
    cat /opt/sglang_commit.txt && \
    echo "Build completed successfully!"

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]