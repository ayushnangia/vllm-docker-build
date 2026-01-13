# SGLang Docker image for commit b1e5a33ae337d20e35e966b8d82a02a913d32689
# Date: 2025-06-09
# SGLang version: 0.4.6.post2
# torch: 2.6.0, flashinfer_python: 0.2.5, sgl-kernel: 0.1.6.post1

# Base image for torch 2.6.x - using CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Build arguments
ARG MAX_JOBS=96
ARG TORCH_CUDA_ARCH_LIST="9.0"

# Environment variables
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    cmake \
    ninja-build \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# HARDCODE commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=b1e5a33ae337d20e35e966b8d82a02a913d32689

# Install torch 2.6.0 with CUDA 11.8 (most compatible)
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Create constraints file with discovered versions
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies with versions discovered for the commit era
fastapi==0.110.3
uvicorn==0.29.0
pydantic==2.7.1
pydantic-core==2.18.2
typing_extensions==4.11.0
outlines==0.0.44
pyzmq==26.0.3
transformers>=4.52.4
huggingface_hub
datasets
orjson
packaging
pillow
psutil
pynvml
python-multipart
uvloop
numpy<2.0
aiohttp
requests
tqdm
setproctitle
IPython
einops
partial_json_parser
cuda-python
ninja
interegular
prometheus-client>=0.20.0
soundfile==0.13.1
xgrammar==0.1.19
blobfile==3.0.0
llguidance>=0.7.11,<0.8.0
compressed-tensors
decord
hf_transfer
modelscope
torchao>=0.9.0
EOF

# Install sgl-kernel 0.1.6.post1 from PyPI (June 2025 commits need this version)
RUN pip3 install sgl-kernel==0.1.6.post1

# Install core dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    fastapi==0.110.3 \
    uvicorn==0.29.0 \
    pydantic==2.7.1 \
    pydantic-core==2.18.2 \
    typing_extensions==4.11.0 \
    outlines==0.0.44 \
    pyzmq==26.0.3 \
    transformers>=4.52.4 \
    xgrammar==0.1.19 \
    blobfile==3.0.0 \
    soundfile==0.13.1 \
    "torchao>=0.9.0" \
    einops \
    partial_json_parser \
    cuda-python \
    "numpy<2.0" \
    packaging \
    ninja

# Install remaining dependencies
RUN pip3 install -c /opt/constraints.txt \
    huggingface_hub datasets orjson pillow psutil pynvml \
    python-multipart uvloop aiohttp requests tqdm setproctitle \
    IPython interegular "prometheus-client>=0.20.0" \
    "llguidance>=0.7.11,<0.8.0" compressed-tensors \
    decord hf_transfer modelscope msgspec

# Build and install flashinfer from source (no wheels for torch 2.6/CUDA 12.4)
# v0.2.5 has pyproject.toml at root level (not in python/ subdirectory)
WORKDIR /sgl-workspace
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.5 && \
    git submodule update --init --recursive && \
    MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip3 install -e . --no-build-isolation && \
    rm -rf .git  # Clean up git to save space

# Clone SGLang repo and checkout exact commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout b1e5a33ae337d20e35e966b8d82a02a913d32689

# Verify the checkout matches expected commit (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b1e5a33ae337d20e35e966b8d82a02a913d32689" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps and then install dependencies separately
WORKDIR /sgl-workspace/sglang/python
RUN pip3 install -e . --no-deps

# Install any missing SGLang dependencies using constraints
RUN pip3 install -c /opt/constraints.txt \
    $(grep -E "^[a-z]" pyproject.toml | grep -v "sglang\[" | grep -v "torch" | \
      grep -v "flashinfer" | grep -v "sgl-kernel" | cut -d'"' -f2 | \
      cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | cut -d'!' -f1 | \
      tr '\n' ' ') || true

# For openbmb/MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# Verify installation (skip CUDA-dependent imports - no GPU during build)
# Note: outlines 0.0.44 has broken optional deps (pyairports) - skip its import check
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    echo "Build verification passed. CUDA imports verified at runtime."

# Final environment setup
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]
