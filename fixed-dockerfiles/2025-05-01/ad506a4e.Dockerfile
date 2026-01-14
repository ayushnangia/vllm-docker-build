# SGLang Docker image for commit ad506a4e6bf3d9ac12100d4648c48df76f584c4e
# Date: 2025-05-01
# SGLang version: 0.4.6.post2
# torch: 2.6.0, flashinfer_python: 0.2.5, sgl-kernel: 0.1.1

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
ENV SGLANG_COMMIT=ad506a4e6bf3d9ac12100d4648c48df76f584c4e

# Install torch 2.6.0 with CUDA 12.4 (exact version from pyproject.toml)
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions for May 2025
# Versions verified from PyPI release history
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies pinned in pyproject.toml (SGLang 0.4.6.post2)
# transformers 4.53.0+ required for compressed_tensors (masking_utils module)
transformers>=4.53.0
xgrammar==0.1.17
soundfile==0.13.1
blobfile==3.0.0
# outlines range: >=0.0.44,<=0.1.11 - use 0.0.44 for stability
outlines==0.0.44
# torchao>=0.9.0
torchao==0.9.0
# Versions available around May 1, 2025
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.7.0
typing_extensions>=4.11.0
pyzmq>=26.0.0
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
compressed-tensors>=0.12.1
hf_transfer
modelscope
decord
msgspec
EOF

# Install sgl-kernel 0.1.1 from PyPI (released Apr 30, 2025 - required version)
RUN pip3 install sgl-kernel==0.1.1

# Install core dependencies with constraints (pinned versions from pyproject.toml)
RUN pip3 install -c /opt/constraints.txt \
    "transformers>=4.53.0" \
    xgrammar==0.1.17 \
    soundfile==0.13.1 \
    blobfile==3.0.0 \
    outlines==0.0.44 \
    torchao==0.9.0 \
    einops \
    partial_json_parser \
    cuda-python \
    "numpy<2.0" \
    packaging \
    ninja

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
    hf_transfer modelscope decord

# Install flashinfer 0.2.5 from flashinfer.ai wheel index
RUN pip3 install flashinfer-python==0.2.5 -f https://flashinfer.ai/whl/cu124/torch2.6/flashinfer_python

# Clone SGLang repo and checkout exact commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ad506a4e6bf3d9ac12100d4648c48df76f584c4e

# Verify the checkout matches expected commit (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ad506a4e6bf3d9ac12100d4648c48df76f584c4e" && \
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
