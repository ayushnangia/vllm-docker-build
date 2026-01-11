# Fixed Dockerfile for SGLang commit a37e1247c183cff86a18f2ed1a075e40704b1c5e
# Date: 2025-07-08
# Target: H100 (CUDA 12.4, compute capability 9.0)

# Base image for torch 2.6.x
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 on Ubuntu 22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 (NOT 2.7.1 due to compatibility issues)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Build flashinfer v0.2.6 from source (no wheel for 0.2.6/0.2.7 with torch 2.6)
RUN pip install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel 0.2.4 from PyPI (confirmed to exist)
RUN pip install sgl-kernel==0.2.4

# HARDCODED commit SHA (1st occurrence)
ENV SGLANG_COMMIT=a37e1247c183cff86a18f2ed1a075e40704b1c5e

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout a37e1247c183cff86a18f2ed1a075e40704b1c5e

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="a37e1247c183cff86a18f2ed1a075e40704b1c5e" && \
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
    sed -i 's/"torchaudio[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao==0.9.0"/"torchao>=0.12.0"/g' python/pyproject.toml && \
    sed -i 's/,,*/,/g; s/,\]/]/g; s/\[,/[/g' python/pyproject.toml

# Install SGLang from source
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[srt]" --extra-index-url https://download.pytorch.org/whl/cu124

# Replace Triton with nightly version (common in SGLang dockerfiles)
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Install additional runtime dependencies
RUN pip install \
    cuda-python \
    einops \
    msgpack \
    datamodel_code_generator

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]