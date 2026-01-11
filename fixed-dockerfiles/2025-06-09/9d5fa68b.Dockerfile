# Fixed Dockerfile for SGLang commit 9d5fa68b (2025-06-09)
# Based on pyproject.toml requirements:
# - torch==2.6.0
# - flashinfer_python==0.2.5 (wheel available)
# - sgl-kernel==0.1.6.post1 (on PyPI)
# - transformers==4.52.3
# - torchao==0.9.0

# Base image for torch 2.6.0 - use CUDA 12.4 and Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    cmake \
    ninja-build \
    software-properties-common \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 on Ubuntu 22.04 (has it built-in)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install build dependencies
RUN pip install ninja numpy packaging

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Pre-install torch 2.6.0 with CUDA 12.4 (BEFORE sglang)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python 0.2.5 from wheel (available for torch 2.6 + CUDA 12.4)
RUN pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel 0.1.6.post1 from PyPI (confirmed available)
RUN pip install sgl-kernel==0.1.6.post1

# Install key dependencies from pyproject.toml
RUN pip install \
    transformers==4.52.3 \
    torchao==0.9.0 \
    xgrammar==0.1.19 \
    "llguidance>=0.7.11,<0.8.0" \
    "outlines>=0.0.44,<=0.1.11" \
    "pyzmq>=25.1.2" \
    "prometheus-client>=0.20.0" \
    blobfile==3.0.0 \
    soundfile==0.13.1

# Install other runtime dependencies
RUN pip install \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    msgspec \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    scipy \
    uvicorn \
    uvloop \
    cuda-python \
    einops \
    openai \
    tiktoken \
    anthropic \
    litellm \
    decord

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=9d5fa68b903d295d2b39201d54905c6801f60f7f

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 9d5fa68b903d295d2b39201d54905c6801f60f7f

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9d5fa68b903d295d2b39201d54905c6801f60f7f" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"xgrammar[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang from source (editable install with all features)
# pyproject.toml is in python/ subdirectory
WORKDIR /sgl-workspace/sglang
RUN cd python && pip install -e ".[all]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Reset to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace
CMD ["/bin/bash"]