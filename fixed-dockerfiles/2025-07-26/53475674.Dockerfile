# SGLang Dockerfile for commit 53475674 (short hash)
# Date: 2025-07-26
# Based on pyproject.toml requirements with torch 2.6.0 (2.7.1 not available for CUDA 12.4)

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from deadsnakes PPA (Ubuntu 22.04)
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 (2.7.1 not available for CUDA 12.4)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Build flashinfer from source (0.2.9rc1 wheel not available)
RUN pip install ninja numpy packaging pybind11 \
    && git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout v0.2.6 \
    && cd python \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . \
    && cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel from PyPI (0.2.7 is available)
RUN pip install sgl-kernel==0.2.7

# HARDCODE the commit SHA (requirement: must appear exactly 3 times)
ENV SGLANG_COMMIT=534756749ae4e664f762de2645a4f63ca2901bab

# Clone SGLang and checkout EXACT commit (hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 534756749ae4e664f762de2645a4f63ca2901bab

# VERIFY commit - compare against HARDCODED value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="534756749ae4e664f762de2645a4f63ca2901bab" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
WORKDIR /sgl-workspace/sglang
RUN sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch==2.7.1",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision==0.22.1",*//g' python/pyproject.toml && \
    sed -i 's/"torchaudio==2.7.1",*//g' python/pyproject.toml && \
    sed -i '/^\s*$/d' python/pyproject.toml

# Patch torchao version (0.9.0 incompatible with torch 2.6.0)
RUN sed -i 's/"torchao==0.9.0"/"torchao>=0.12.0"/' python/pyproject.toml

# Install SGLang from source with all dependencies
RUN cd python && pip install -e ".[srt]"

# Install Triton nightly (often needed for latest features)
RUN pip uninstall -y triton triton-nightly || true \
    && pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'Torch {torch.__version__} with CUDA OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]