# Fixed Dockerfile for SGLang commit db452760e5b2378efd06b1ceb9385d2eeb6d217c
# Date: 2025-04-07
# torch==2.5.1, flashinfer_python==0.2.3, sgl-kernel==0.0.8 (from source), transformers==4.51.0

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    ca-certificates \
    ccache \
    cmake \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
RUN apt-get update && apt-get install -y \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
    && wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 \
    && cd .. && rm -rf Python-3.10.14*

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install torch 2.5.1 with CUDA 12.1
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer_python==0.2.3 from pre-built wheel
RUN pip3 install flashinfer-python==0.2.3 -i https://flashinfer.ai/whl/cu121/torch2.5/flashinfer-python/

# Build sgl-kernel from source (version 0.0.8 not on PyPI)
RUN pip3 install ninja packaging && \
    git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel && \
    cd /tmp/sgl-kernel && \
    git checkout v0.0.8 && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    rm -rf /tmp/sgl-kernel

# Install other key dependencies
RUN pip3 install transformers==4.51.0

# HARDCODE the commit SHA (exact commit required for benchmarking)
ENV SGLANG_COMMIT=db452760e5b2378efd06b1ceb9385d2eeb6d217c

# Clone SGLang and checkout EXACT commit (SHA is hardcoded, not variable)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout db452760e5b2378efd06b1ceb9385d2eeb6d217c

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="db452760e5b2378efd06b1ceb9385d2eeb6d217c" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install additional dependencies from pyproject.toml
RUN pip3 install \
    torchao>=0.7.0 \
    xgrammar==0.1.17 \
    compressed-tensors \
    datasets \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    "llguidance>=0.6.15" \
    modelscope \
    ninja \
    orjson \
    packaging \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    "pyzmq>=25.1.2" \
    "soundfile==0.13.1" \
    uvicorn \
    uvloop \
    cuda-python \
    "outlines>=0.0.44,<=0.1.11" \
    partial_json_parser \
    einops

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl_kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace

# Entrypoint for SGLang server
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]