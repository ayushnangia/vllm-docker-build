# Fixed Dockerfile for SGLang commit 33b242df303e03886835d08a583fefe979a3ee88
# Date: 2024-05-11
# Based on discovered versions from PyPI for May 2024 era

# Base image for torch 2.3.0 with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set build environment variables
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV CUDA_HOME=/usr/local/cuda

# 1st hardcoded commit SHA: ENV variable
ENV SGLANG_COMMIT=33b242df303e03886835d08a583fefe979a3ee88

# Create workspace
WORKDIR /sgl-workspace

# Create constraints file with discovered versions from PyPI (May 2024 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-05-11 era
# All versions verified via WebFetch from PyPI
fastapi==0.111.0    # May 3, 2024
uvicorn==0.29.0     # March 20, 2024
pydantic==2.7.1     # April 23, 2024 (v2 required by outlines)
typing_extensions==4.11.0  # April 5, 2024 (before 4.14 to avoid Sentinel)
outlines==0.0.34    # vLLM 0.4.2 pins this version
pyzmq==26.0.3       # May 1, 2024
tiktoken==0.6.0
lm-format-enforcer==0.9.8
transformers==4.40.2
tokenizers==0.19.1
sentencepiece==0.2.0
prometheus_client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
py-cpuinfo==9.0.0
psutil==5.9.8
numpy==1.26.4
requests==2.31.0
filelock==3.14.0
nvidia-ml-py==12.550.107
ray==2.23.0
aiohttp==3.9.5
uvloop==0.19.0
rpyc==6.0.0
pillow==10.3.0
interegular==0.3.3
packaging==24.0
EOF

# Install vLLM 0.4.2 with --no-deps first
RUN pip install --upgrade pip && \
    pip install vllm==0.4.2 --no-deps

# Install vLLM dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    transformers \
    tokenizers \
    fastapi \
    openai \
    "uvicorn[standard]" \
    pydantic \
    prometheus_client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing_extensions \
    filelock \
    ray \
    nvidia-ml-py \
    "vllm-nccl-cu12>=2.18,<2.19"

# Install xformers with --no-deps to prevent pulling wrong torch version
RUN pip install xformers==0.0.26.post1 --no-deps

# Install flashinfer 0.0.8 from flashinfer.ai wheels (available for torch 2.3)
RUN pip install flashinfer==0.0.8 -i https://flashinfer.ai/whl/cu121/torch2.3/

# 2nd hardcoded commit SHA: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 33b242df303e03886835d08a583fefe979a3ee88 && \
    cd ..

# 3rd hardcoded commit SHA: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="33b242df303e03886835d08a583fefe979a3ee88" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL"; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies using constraints
# NOTE: Do NOT reinstall torch - keep the 2.3.0 from base image
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    uvloop \
    uvicorn \
    pyzmq \
    interegular \
    pydantic \
    pillow \
    outlines \
    packaging \
    requests \
    tqdm \
    openai \
    "numpy<2.0" \
    tiktoken \
    anthropic

# Install datasets (from original Dockerfile)
RUN pip install datasets

# Verify installations
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
RUN pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python -c "import sglang; print('SGLang import successful')" && \
    python -c "import outlines; print('Outlines import successful')" && \
    python -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python -c "import flashinfer; print('Flashinfer import successful')"

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]