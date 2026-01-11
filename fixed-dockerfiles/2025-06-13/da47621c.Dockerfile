# Fixed Dockerfile for SGLang commit (2025-06-13)
# Date: 2025-06-13
# SGLang version: 0.4.7
# Torch: 2.7.1 (available with CUDA 12.6)
# Flashinfer: 0.2.6.post1 (built from source, no wheel for torch 2.7)
# sgl-kernel: 0.1.7 (from PyPI)
# CUDA: 12.6

FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from deadsnakes PPA (Ubuntu 22.04)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.7.1 with CUDA 12.6 support
RUN pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install ninja for building flashinfer
RUN pip3 install ninja numpy packaging

# Build flashinfer 0.2.6.post1 from source (no wheel for torch 2.7)
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6.post1 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    cd / && \
    rm -rf /tmp/flashinfer

# Install sgl-kernel 0.1.7 from PyPI (verified available)
RUN pip3 install sgl-kernel==0.1.7

# HARDCODE commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=da47621ccc4f8e8381f3249257489d5fe32aff1b

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout da47621ccc4f8e8381f3249257489d5fe32aff1b

# VERIFY commit matches expected SHA (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="da47621ccc4f8e8381f3249257489d5fe32aff1b" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install xformers compatible with torch 2.7
RUN pip3 install xformers==0.0.33.post2 --index-url https://download.pytorch.org/whl/cu126

# Install transformers explicitly to ensure correct version
RUN pip3 install transformers==4.52.3

# Install torchao
RUN pip3 install torchao==0.9.0

# Install xgrammar
RUN pip3 install xgrammar==0.1.19

# Install SGLang from source (editable install)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Replace Triton with nightly version (common fix for compatibility)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit SHA file exists at /opt/sglang_commit.txt"

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]