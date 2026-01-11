# Fixed Dockerfile for SGLang commit 3212c2ad3f7e4fb473dc807b4b176020a778ed5b
# Date: 2025-07-26
# torch 2.7.1 from cu128 index (as specified in pyproject.toml)
# flashinfer_python 0.2.9rc1 built from source (no wheel available)
# sgl-kernel 0.2.7 from PyPI

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential cmake ninja-build \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.7.1 from CUDA 12.8 index (required for this version)
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Build flashinfer_python 0.2.9rc1 from source (no wheel available for this version)
RUN pip install ninja numpy packaging \
    && git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout v0.2.9rc1 \
    && cd python \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . \
    && rm -rf /tmp/flashinfer

# Install sgl-kernel 0.2.7 from PyPI (verified available)
RUN pip install sgl-kernel==0.2.7

# Install transformers with correct version
RUN pip install transformers==4.53.2

# Install torchao with version specified in pyproject.toml
RUN pip install torchao==0.9.0

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=3212c2ad3f7e4fb473dc807b4b176020a778ed5b

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 3212c2ad3f7e4fb473dc807b4b176020a778ed5b

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="3212c2ad3f7e4fb473dc807b4b176020a778ed5b" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    if [ -f python/pyproject.toml ]; then \
        sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"torchaudio[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/,,*/,/g' python/pyproject.toml && \
        sed -i 's/\[,/\[/g' python/pyproject.toml && \
        sed -i 's/,\]/\]/g' python/pyproject.toml; \
    fi

# Install other runtime dependencies
RUN pip install \
    aiohttp requests tqdm numpy IPython setproctitle \
    blobfile==3.0.0 build compressed-tensors datasets fastapi hf_transfer \
    huggingface_hub interegular "llguidance>=0.7.11,<0.8.0" modelscope msgspec \
    orjson outlines==0.1.11 packaging partial_json_parser pillow \
    "prometheus-client>=0.20.0" psutil pydantic pynvml pybase64 \
    python-multipart "pyzmq>=25.1.2" sentencepiece soundfile==0.13.1 scipy \
    timm==1.0.16 uvicorn uvloop xgrammar==0.1.21 \
    cuda-python einops

# Install SGLang from checked-out source
WORKDIR /sgl-workspace/sglang
RUN cd python && pip install -e .

# Install Triton nightly (common pattern in SGLang Dockerfiles)
RUN pip uninstall -y triton triton-nightly || true \
    && pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'Torch {torch.__version__} with CUDA OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]