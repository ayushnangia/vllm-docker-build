# Fixed Dockerfile for SGLang commit a191a0e4
# Date: 2025-05-26
# torch: 2.6.0, flashinfer_python: 0.2.5, sgl-kernel: 0.1.4

# Base image for torch 2.6.x (requires CUDA 12.4)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 for Ubuntu 22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 with CUDA 12.4
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python 0.2.5 from wheel (available for torch 2.6 + CUDA 12.4)
# Note: The package name in the index uses underscore
RUN pip3 install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel 0.1.4 from PyPI (confirmed available)
RUN pip3 install sgl-kernel==0.1.4

# Install other key dependencies from pyproject.toml
RUN pip3 install \
    transformers==4.51.1 \
    torchao==0.9.0 \
    xgrammar==0.1.19

# HARDCODE the commit SHA (1st occurrence of 3)
ENV SGLANG_COMMIT=a191a0e47c2f0b0c8aed28080b9cb78624365e92

# Clone SGLang and checkout EXACT commit (2nd occurrence of 3 - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout a191a0e47c2f0b0c8aed28080b9cb78624365e92

# VERIFY commit - compare against HARDCODED value (3rd occurrence of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="a191a0e47c2f0b0c8aed28080b9cb78624365e92" && \
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
    sed -i 's/,,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang from source in editable mode (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install additional runtime dependencies that might be needed
RUN pip3 install \
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
    ninja \
    orjson \
    packaging \
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
    uvicorn \
    uvloop \
    cuda-python \
    "outlines>=0.0.44,<=0.1.11" \
    einops

# Set environment for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]