# SGLang Dockerfile for commit ad506a4e6bf3d9ac12100d4648c48df76f584c4e (2025-05-01)
# Base image for torch 2.6.0 - using CUDA 12.4.1 with Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
# HARDCODE the commit SHA - this Dockerfile is specific to this exact commit
ENV SGLANG_COMMIT=ad506a4e6bf3d9ac12100d4648c48df76f584c4e

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (Ubuntu 22.04 has it by default, but ensure pip is installed)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 with CUDA 12.4 index
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python 0.2.5 from flashinfer.ai wheels (verified available)
RUN pip3 install flashinfer_python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel 0.1.1 from PyPI (verified available)
RUN pip3 install sgl-kernel==0.1.1

# Install transformers with the pinned version
RUN pip3 install transformers==4.51.1

# Set working directory
WORKDIR /sgl-workspace

# Clone SGLang repository and checkout the EXACT commit (HARDCODED SHA)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ad506a4e6bf3d9ac12100d4648c48df76f584c4e

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ad506a4e6bf3d9ac12100d4648c48df76f584c4e" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
# Remove flashinfer_python, sgl-kernel, torch, torchvision, and transformers since we pre-installed them
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    # Clean up any double commas or trailing commas
    sed -i 's/,,/,/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]" --no-deps && \
    pip3 install -e "python[runtime_common]" --no-deps && \
    pip3 install \
        aiohttp \
        requests \
        tqdm \
        numpy \
        IPython \
        setproctitle \
        compressed-tensors \
        datasets \
        decord \
        fastapi \
        hf_transfer \
        huggingface_hub \
        interegular \
        "llguidance>=0.7.11,<0.8.0" \
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
        "torchao>=0.9.0" \
        uvicorn \
        uvloop \
        "xgrammar==0.1.17" \
        "blobfile==3.0.0" \
        cuda-python \
        "outlines>=0.0.44,<=0.1.11" \
        partial_json_parser \
        einops

# Install datamodel_code_generator for MiniCPM models
RUN pip3 install datamodel_code_generator

# Verify SGLang installation
RUN python3 -c "import sglang; print('SGLang import OK')"

# Verify flashinfer installation
RUN python3 -c "import flashinfer; print('Flashinfer import OK')"

# Verify torch CUDA availability
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Set environment for interactive use
ENV DEBIAN_FRONTEND=interactive

# Set TORCH_CUDA_ARCH_LIST for H100 (compute capability 9.0)
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
