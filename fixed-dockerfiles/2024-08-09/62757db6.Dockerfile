# Fixed Dockerfile for SGLang commit from August 9, 2024
# Base image for torch 2.4.0 with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.1
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# 1st SHA occurrence - ENV
ENV SGLANG_COMMIT=62757db6f0f09a6dff15b1ee1ac3029602951509

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
        git \
        curl \
        wget \
        vim \
        tzdata \
        cmake \
        ninja-build \
        build-essential \
        software-properties-common \
        libnuma-dev \
        ccache && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (since deadsnakes PPA deprecated for Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install -y \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd .. && \
    rm -rf Python-3.10.14 Python-3.10.14.tgz && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.4.0 (required by vLLM 0.5.4)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI (August 2024 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-08-09 era
fastapi==0.112.0
uvicorn==0.30.5
pydantic==2.8.2
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.1.0
aiohttp==3.10.1
huggingface_hub==0.24.5
pillow==10.4.0
psutil==6.0.0
ray==2.9.3
EOF

# Install vLLM 0.5.4 without dependencies first
RUN pip install vllm==0.5.4 --no-deps

# Install vLLM dependencies from requirements (using constraints where applicable)
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    fastapi \
    aiohttp \
    openai \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.3" \
    "outlines>=0.0.43,<0.1" \
    typing_extensions \
    "filelock>=3.10.4" \
    pyzmq \
    "ray>=2.9" \
    nvidia-ml-py \
    xformers==0.0.27.post2 \
    vllm-flash-attn==2.6.1

# 2nd SHA occurrence - git checkout
RUN git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git checkout 62757db6f0f09a6dff15b1ee1ac3029602951509

# 3rd SHA occurrence - verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="62757db6f0f09a6dff15b1ee1ac3029602951509" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install flashinfer from pre-built wheels
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ || \
    (echo "flashinfer wheel not found, building from source..." && \
     git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
     cd /tmp/flashinfer/python && \
     pip install . && \
     cd / && \
     rm -rf /tmp/flashinfer)

# Build and install sgl-kernel from source (not available on PyPI in August 2024)
RUN cd /sgl-workspace/sglang && \
    if [ -d "sgl-kernel" ]; then \
        cd sgl-kernel && \
        pip install -e .; \
    else \
        echo "sgl-kernel directory not found, skipping..."; \
    fi

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    uvicorn \
    uvloop \
    pyzmq

# Install triton-nightly to avoid version conflicts
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" && \
    python3 -c "import sglang; print(f'SGLang version: {sglang.__version__}')" && \
    python3 -c "import outlines; print(f'Outlines version: {outlines.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    echo "All imports successful!"

# Set environment for runtime
ENV DEBIAN_FRONTEND=interactive
ENV HF_HUB_OFFLINE=0
ENV HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /sgl-workspace/sglang

# Add label for commit tracking (SHA in ENV for verification)

# Default entrypoint
ENTRYPOINT ["/bin/bash"]