# Docker image for SGLang commit b170930534acbb9c1619a3c83670a839ceee763a (2024-10-17)
# Based on discovered dependencies from PyPI and repo analysis

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.1

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    software-properties-common \
    ninja-build \
    libibverbs-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-dev python3.10-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && curl https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel packaging

# Install PyTorch 2.4.0 and related packages (required by vLLM 0.5.5)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install specific versions of xformers and vllm-flash-attn
RUN pip install xformers==0.0.27.post2 vllm-flash-attn==2.6.1

# Create constraints file with discovered versions from PyPI (October 2024 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies with versions discovered from PyPI for 2024-10-17 era
fastapi==0.115.2
uvicorn==0.32.0
pydantic==2.9.2
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
# Additional pinned versions for stability
aiohttp==3.10.5
pillow==10.4.0
psutil==6.0.0
prometheus_client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
orjson==3.10.7
msgspec==0.18.6
EOF

# Install vLLM 0.5.5 without dependencies
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    'numpy<2.0.0' \
    requests \
    tqdm \
    py-cpuinfo \
    'transformers>=4.43.2' \
    'tokenizers>=0.19.1' \
    protobuf \
    fastapi \
    aiohttp \
    'openai>=1.0' \
    'uvicorn[standard]' \
    'pydantic>=2.8' \
    pillow \
    'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' \
    'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.6' \
    'outlines>=0.0.43,<0.1' \
    'typing_extensions>=4.10' \
    'filelock>=3.10.4' \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    'gguf==0.9.1' \
    importlib_metadata \
    ray \
    nvidia-ml-py

# Set environment for build
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Build flashinfer from source (no wheels for cu121/torch2.4)
WORKDIR /tmp
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.1.6 && \
    cd python && \
    pip install . && \
    cd / && rm -rf /tmp/flashinfer

WORKDIR /sgl-workspace

# FIRST HARDCODED SHA: Environment variable
ENV SGLANG_COMMIT=b170930534acbb9c1619a3c83670a839ceee763a

# SECOND HARDCODED SHA: Clone and checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout b170930534acbb9c1619a3c83670a839ceee763a

# THIRD HARDCODED SHA: Verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b170930534acbb9c1619a3c83670a839ceee763a" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python && pip install -e . --no-deps

# Install SGLang dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    pyzmq \
    'outlines>=0.0.44' \
    modelscope

# For openbmb/MiniCPM models
RUN pip install datamodel_code_generator

# Final sanity check
RUN python3 -c "import sglang; import vllm; import outlines; print('All imports successful')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import typing_extensions; print(f'typing_extensions version: {typing_extensions.__version__}')" && \
    cat /opt/sglang_commit.txt

# Set working directory
WORKDIR /sgl-workspace

# Label for tracking
LABEL org.opencontainers.image.revision=b170930534acbb9c1619a3c83670a839ceee763a

ENV DEBIAN_FRONTEND=interactive