# Fixed Dockerfile for SGLang commit cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a
# Date: 2025-05-08
# torch==2.6.0, flashinfer_python==0.2.5, sgl-kernel==0.1.1

# Base image for torch 2.6.0 - use CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# HARDCODED COMMIT SHA - occurrence 1/3
ENV SGLANG_COMMIT=cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a

# Pre-install torch with CUDA 12.4 index (version from pyproject.toml)
RUN python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install torchvision to match torch version
RUN python3 -m pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer from wheel (available for torch 2.6 + CUDA 12.4)
RUN python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python

# Install sgl-kernel from PyPI (version 0.1.1 exists)
RUN python3 -m pip install sgl-kernel==0.1.1

# Install other critical dependencies before SGLang
RUN python3 -m pip install transformers==4.51.1 \
    && python3 -m pip install "torchao>=0.9.0" \
    && python3 -m pip install cuda-python \
    && python3 -m pip install numpy packaging ninja

# Clone SGLang and checkout EXACT commit - HARDCODED SHA occurrence 2/3
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a

# VERIFY commit matches expected - HARDCODED SHA occurrence 3/3
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN python3 -m pip install -e "python[all]"

# Install triton-nightly (common pattern in SGLang Dockerfiles)
RUN python3 -m pip uninstall -y triton triton-nightly || true \
    && python3 -m pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    COMMIT=$(cat /opt/sglang_commit.txt) && \
    echo "Docker image built for SGLang commit: $COMMIT"

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Set entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]