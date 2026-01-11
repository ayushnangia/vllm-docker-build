# Base image for torch 2.7.1 with CUDA 12.4
# SGLang Dockerfile for commit 777688b8929c877e4e28c2eac208d776abe4c3af
# Date: 2025-06-11
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=777688b8929c877e4e28c2eac208d776abe4c3af

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    libibverbs-dev \
    rdma-core \
    infiniband-diags \
    openssh-server \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (Ubuntu 22.04 has Python 3.10 by default)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.7.1 with CUDA 12.6 index (torch 2.7.1 requires newer CUDA index)
RUN pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install torchvision and torchaudio compatible with torch 2.7.1
RUN pip3 install torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Build flashinfer from source (no wheels available for torch 2.7)
RUN pip3 install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# Install cuda-python
RUN pip3 install cuda-python

# Install sgl-kernel 0.1.7 from PyPI (verified available)
RUN pip3 install sgl-kernel==0.1.7

# Install other dependencies before SGLang
RUN pip3 install transformers==4.52.3 \
    torchao==0.9.0 \
    xgrammar==0.1.19 \
    einops \
    outlines>=0.0.44,<=0.1.11

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 777688b8929c877e4e28c2eac208d776abe4c3af

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="777688b8929c877e4e28c2eac208d776abe4c3af" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
WORKDIR /sgl-workspace/sglang
RUN cd python && \
    sed -i 's/"flashinfer[^"]*",*//g' pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' pyproject.toml && \
    sed -i '/^[[:space:]]*,$/d' pyproject.toml && \
    sed -i 's/,]/]/g' pyproject.toml && \
    sed -i 's/,,/,/g' pyproject.toml

# Install SGLang from source (editable install)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install additional runtime dependencies
RUN pip3 install msgspec aiohttp requests tqdm numpy IPython setproctitle \
    fastapi uvicorn uvloop pydantic "prometheus-client>=0.20.0" psutil \
    python-multipart "pyzmq>=25.1.2" partial_json_parser pillow \
    orjson packaging hf_transfer huggingface_hub \
    compressed-tensors datasets interegular modelscope \
    "blobfile==3.0.0" "soundfile==0.13.1" scipy \
    "llguidance>=0.7.11,<0.8.0" pynvml ninja

# Install optional dependencies for OpenAI, Anthropic, etc.
RUN pip3 install "openai>=1.0" tiktoken "anthropic>=0.20.0" \
    "litellm>=1.0.0" "torch_memory_saver>=0.0.4" decord

# Replace Triton with nightly version if needed
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]