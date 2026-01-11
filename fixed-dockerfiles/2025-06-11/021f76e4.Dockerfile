# Dockerfile for SGLang commit 021f76e4f49861b2e9ea9ccff06a46d577e3c548
# Date: 2025-06-11
# torch 2.6.0 + CUDA 12.4

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
    libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    liblzma-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 on Ubuntu 22.04
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 with CUDA 12.4
RUN pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install ninja for building
RUN pip3 install ninja numpy packaging

# Build flashinfer from source (v0.2.6) since wheel doesn't exist for torch 2.6 + CUDA 12.4
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel from PyPI (0.1.7 is available)
RUN pip3 install sgl-kernel==0.1.7

# Install other dependencies that SGLang needs
RUN pip3 install \
    transformers==4.52.3 \
    "outlines>=0.0.44,<=0.1.11" \
    einops \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    "llguidance>=0.7.11,<0.8.0" \
    modelscope \
    msgspec \
    orjson \
    partial_json_parser \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    "pyzmq>=25.1.2" \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19 \
    cuda-python

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=021f76e4f49861b2e9ea9ccff06a46d577e3c548

# Clone SGLang repo and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 021f76e4f49861b2e9ea9ccff06a46d577e3c548

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="021f76e4f49861b2e9ea9ccff06a46d577e3c548" && \
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
    sed -i 's/"torchaudio[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Set working directory for runtime
WORKDIR /sgl-workspace/sglang

# Entry point
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
