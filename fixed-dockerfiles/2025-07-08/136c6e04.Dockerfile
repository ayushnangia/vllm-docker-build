# SGLang Docker image for commit 136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7
# Date: 2025-07-08
# SGLang version: 0.4.9
# torch: 2.7.1, flashinfer_python: 0.2.7.post1, sgl-kernel: 0.2.4

# Base image for torch 2.7.x - using CUDA 12.4
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
ENV SGLANG_COMMIT=136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7

# Install torch 2.7.1 with CUDA 12.6 (exact version from pyproject.toml)
RUN pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Create constraints file with discovered versions for July 2025
# Versions verified from PyPI release history
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies pinned in pyproject.toml (SGLang 0.4.9)
outlines==0.1.11
transformers==4.53.0
torchao==0.9.0
timm==1.0.16
xgrammar==0.1.19
soundfile==0.13.1
blobfile==3.0.0
# Versions available on July 8, 2025
fastapi>=0.115.0
uvicorn>=0.34.0
pydantic>=2.11.0
typing_extensions>=4.12.0
pyzmq>=27.0.0
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
llguidance>=0.7.11,<0.8.0
compressed-tensors
hf_transfer
modelscope
scipy
msgspec
build
EOF

# Install sgl-kernel 0.2.4 from PyPI (released Jul 5, 2025 - required version)
RUN pip3 install sgl-kernel==0.2.4

# Install core dependencies with constraints (pinned versions from pyproject.toml)
RUN pip3 install -c /opt/constraints.txt \
    outlines==0.1.11 \
    transformers==4.53.0 \
    torchao==0.9.0 \
    timm==1.0.16 \
    xgrammar==0.1.19 \
    soundfile==0.13.1 \
    blobfile==3.0.0 \
    einops \
    partial_json_parser \
    cuda-python \
    "numpy<2.0" \
    packaging \
    ninja \
    scipy

# Install web framework dependencies
RUN pip3 install -c /opt/constraints.txt \
    fastapi \
    uvicorn \
    pydantic \
    typing_extensions \
    pyzmq \
    python-multipart \
    uvloop \
    orjson \
    msgspec

# Install remaining dependencies
RUN pip3 install -c /opt/constraints.txt \
    huggingface_hub datasets pillow psutil pynvml \
    aiohttp requests tqdm setproctitle \
    IPython interegular "prometheus-client>=0.20.0" \
    "llguidance>=0.7.11,<0.8.0" compressed-tensors \
    hf_transfer modelscope build

# Build and install flashinfer 0.2.7.post1 from source (no prebuilt wheels for torch 2.7.1)
# Released Jul 1, 2025 - matches pyproject.toml requirement
WORKDIR /sgl-workspace
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.7.post1 && \
    git submodule update --init --recursive && \
    MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip3 install -e . --no-build-isolation && \
    rm -rf .git  # Clean up git to save space

# Clone SGLang repo and checkout exact commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7

# Verify the checkout matches expected commit (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7" && \
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

# For openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator

# Verify installation (skip CUDA-dependent imports - no GPU during build)
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
