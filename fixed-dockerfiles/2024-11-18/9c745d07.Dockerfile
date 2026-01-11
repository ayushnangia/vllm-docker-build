# Fixed Dockerfile for SGLang commit 9c745d078e29e153a64300bd07636c7c9c1c42d5
# Date: 2024-11-18
# Versions discovered from PyPI for this commit date

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Build arguments
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG MAX_JOBS=96
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
ENV MAX_JOBS=$MAX_JOBS
ENV DEBIAN_FRONTEND=noninteractive

# First occurrence of commit SHA - ENV
ENV SGLANG_COMMIT=9c745d078e29e153a64300bd07636c7c9c1c42d5

# Set timezone to avoid interactive prompts
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        curl \
        git \
        sudo \
        wget \
        vim \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        libjpeg-dev \
        libpng-dev \
        libibverbs-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (Ubuntu 20.04 deadsnakes PPA is broken)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    cd .. && rm -rf Python-3.10.14*

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 with CUDA 12.1 (required by vLLM 0.6.3.post1)
RUN pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with versions discovered from PyPI for 2024-11-18 era
# These versions are carefully chosen to avoid conflicts
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies with versions from November 2024
# Discovered via PyPI release history checks
fastapi==0.115.5
uvicorn==0.32.0
pydantic==2.9.2
pydantic-core==2.23.4
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
aiohttp==3.10.10
requests==2.32.3
numpy==1.26.4
setuptools==75.3.0
wheel==0.44.0
EOF

# Install vLLM 0.6.3.post1 without dependencies first
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with constraints
# These are from vLLM's requirements-common.txt and requirements-cuda.txt
RUN pip3 install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    "requests>=2.26.0" \
    tqdm \
    py-cpuinfo \
    "transformers>=4.45.2" \
    "tokenizers>=0.19.1" \
    protobuf \
    "fastapi>=0.107.0,!=0.113.*,!=0.114.0" \
    aiohttp \
    "openai>=1.40.0" \
    "uvicorn[standard]" \
    "pydantic>=2.9" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    partial-json-parser \
    pyzmq \
    msgspec \
    "gguf==0.10.0" \
    importlib_metadata \
    "mistral_common[opencv]>=1.4.4" \
    pyyaml \
    six \
    einops \
    "compressed-tensors==0.6.0" \
    "ray>=2.9" \
    nvidia-ml-py \
    "xformers==0.0.27.post2"

# Set working directory
WORKDIR /sgl-workspace

# Second occurrence of commit SHA - git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 9c745d078e29e153a64300bd07636c7c9c1c42d5

# Third occurrence of commit SHA - verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9c745d078e29e153a64300bd07636c7c9c1c42d5" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Build and install flashinfer from source (no prebuilt wheels for torch 2.4.0)
RUN cd /tmp && \
    git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.1.6 && \
    cd python && \
    pip3 install ninja && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install . && \
    cd / && rm -rf /tmp/flashinfer

# Install SGLang without dependencies first
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang dependencies with constraints
# These are from sglang pyproject.toml runtime_common and srt dependencies
RUN pip3 install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    IPython \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    "pyzmq>=25.1.2" \
    "outlines>=0.0.44,<0.1.0" \
    modelscope

# Install sgl-kernel if available
RUN pip3 install sgl-kernel || echo "sgl-kernel not available, continuing without it"

# For openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator

# Replace Triton with nightly version (common pattern in SGLang dockerfiles)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Clear pip cache
RUN pip3 cache purge

# Final verification
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('FlashInfer imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')"

# Add labels for container metadata
LABEL org.opencontainers.image.revision="9c745d078e29e153a64300bd07636c7c9c1c42d5"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.description="SGLang Docker image for commit 9c745d078e29e153a64300bd07636c7c9c1c42d5"

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]