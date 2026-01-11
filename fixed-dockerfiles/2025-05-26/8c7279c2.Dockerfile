# SGLang Dockerfile for commit 8c7279c24e535681478188967b3007916b87b3d0
# Date: 2025-05-26
# torch 2.6.0 requires CUDA 12.4 base image

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (Ubuntu 22.04 has it by default, but ensure proper setup)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch with CUDA 12.4 index (MUST be done before sglang)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install sgl-kernel from PyPI (version 0.1.4 is available)
RUN pip install sgl-kernel==0.1.4

# Try to install flashinfer from flashinfer.ai website, fallback to build from source
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6/ || \
    (echo "Flashinfer wheel not found, building from source..." && \
     pip install ninja numpy packaging && \
     git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
     cd /tmp/flashinfer && \
     git checkout v0.2.5 && \
     cd python && \
     TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip install --no-build-isolation . && \
     rm -rf /tmp/flashinfer)

# Install other specific dependencies from pyproject.toml
RUN pip install transformers==4.51.1 torchao==0.9.0

# HARDCODE the commit SHA (don't use ARG to avoid forgotten --build-arg issues)
ENV SGLANG_COMMIT=8c7279c24e535681478188967b3007916b87b3d0

# Clone SGLang and checkout EXACT commit (hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 8c7279c24e535681478188967b3007916b87b3d0

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="8c7279c24e535681478188967b3007916b87b3d0" && \
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
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[srt]"

# Install additional dependencies from runtime_common if not already installed
RUN pip install \
    "blobfile==3.0.0" \
    "compressed-tensors" \
    "datasets" \
    "fastapi" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "llguidance>=0.7.11,<0.8.0" \
    "modelscope" \
    "msgspec" \
    "ninja" \
    "orjson" \
    "packaging" \
    "partial_json_parser" \
    "pillow" \
    "prometheus-client>=0.20.0" \
    "psutil" \
    "pydantic" \
    "pynvml" \
    "python-multipart" \
    "pyzmq>=25.1.2" \
    "soundfile==0.13.1" \
    "scipy" \
    "uvicorn" \
    "uvloop" \
    "xgrammar==0.1.19" \
    "cuda-python" \
    "outlines>=0.0.44,<=0.1.11" \
    "einops"

# For H100 GPU optimization
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')"
RUN python3 -c "import flashinfer; print('Flashinfer import OK')"
RUN python3 -c "import sgl_kernel; print('sgl-kernel import OK')"
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}')"
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Set working directory for runtime
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]