# SGLang Docker image
# Date: 2025-07-26
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set build arguments and environment variables
ARG CUDA_VERSION=12.4
ARG PYTHON_VERSION=3.10
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG MAX_JOBS=96

# 1st occurrence: ENV with commit SHA (HARDCODED)
ENV SGLANG_COMMIT=534756749ae4e664f762de2645a4f63ca2901bab
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl build-essential cmake ninja-build \
    libssl-dev libffi-dev \
    ccache \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.7.1 with CUDA 12.4 support
RUN pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from PyPI for July 2025 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-07-26 era
fastapi==0.116.1
uvicorn==0.35.0
pydantic==2.11.7
typing_extensions==4.13.2
outlines==0.1.11
pyzmq==27.0.0
# Core outlines dependency
outlines-core==0.1.26
# Other common versions from the era
numpy<2
msgspec
orjson
aiohttp
requests
tqdm
cloudpickle
diskcache
interegular
jinja2
lark
nest_asyncio
referencing
jsonschema
pycountry
airportsdata
EOF

# Install runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    blobfile==3.0.0 \
    build \
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
    outlines==0.1.11 \
    packaging \
    partial_json_parser \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    pynvml \
    pybase64 \
    python-multipart \
    "pyzmq>=25.1.2" \
    sentencepiece \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    transformers==4.53.2 \
    timm==1.0.16 \
    uvicorn \
    uvloop \
    xgrammar==0.1.21

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.2.7

# Install other srt dependencies
RUN pip install cuda-python einops

# Create workspace directory
WORKDIR /sgl-workspace

# 2nd occurrence: git clone and checkout with exact commit SHA
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 534756749ae4e664f762de2645a4f63ca2901bab

# 3rd occurrence: verification of exact commit SHA
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="534756749ae4e664f762de2645a4f63ca2901bab" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Build and install flashinfer from source since wheel for 0.2.9rc1 is not available
RUN cd /tmp && \
    git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.9rc1 || git checkout 0.2.9rc1 || \
    (echo "Warning: Could not find exact tag, using closest commit" && \
     git log --oneline | head -5) && \
    cd python && \
    pip install . --no-deps && \
    cd / && rm -rf /tmp/flashinfer

# Install SGLang in editable mode with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install core SGLang dependencies that weren't in runtime_common
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Final verification
RUN python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('FlashInfer import successful')" && \
    echo "Build completed successfully"

# Set working directory to SGLang
WORKDIR /sgl-workspace/sglang

# Label the image with the commit
LABEL org.opencontainers.image.revision=${SGLANG_COMMIT}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.description="SGLang Docker image for commit ${SGLANG_COMMIT}"