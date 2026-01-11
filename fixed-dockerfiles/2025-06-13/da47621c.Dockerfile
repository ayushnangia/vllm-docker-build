# Fixed Dockerfile for SGLang commit da47621ccc4f8e8381f3249257489d5fe32aff1b
# Date: 2025-06-13
# SGLang version: 0.4.7
# Torch: 2.7.1 (with CUDA 12.4)
# vLLM: 0.9.1 (discovered via PyPI for June 2025)
# Flashinfer: 0.2.6.post1 (built from source, no wheel available)
# sgl-kernel: 0.1.7 (from PyPI)

# Base image for torch 2.7.x (requires CUDA 12.4+)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Build arguments and environment variables
ARG TORCH_VERSION=2.7.1
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# 1st SHA occurrence: ENV (REQUIRED)
ENV SGLANG_COMMIT=da47621ccc4f8e8381f3249257489d5fe32aff1b

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.10 is default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.7.1 with CUDA 12.4
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from PyPI (June 2025 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-06-13 era
fastapi==0.115.12
uvicorn==0.34.3
pydantic==2.11.6
typing_extensions==4.12.2
outlines==0.1.11
pyzmq==27.0.0
transformers==4.52.3
huggingface-hub==0.32.0
tokenizers==0.21.1
prometheus-client==0.20.0
EOF

# Install vLLM 0.9.1 with --no-deps to avoid dependency conflicts
RUN pip install vllm==0.9.1 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    fastapi[standard] \
    pydantic \
    openai \
    aiohttp \
    transformers \
    tokenizers \
    huggingface-hub[hf_xet] \
    compressed-tensors==0.10.1 \
    mistral_common[opencv] \
    gguf \
    einops \
    prometheus_client \
    prometheus-fastapi-instrumentator \
    opentelemetry-sdk \
    opentelemetry-api \
    outlines \
    lm-format-enforcer \
    ray[cgraph] \
    numpy \
    regex \
    requests \
    tiktoken \
    pillow \
    protobuf \
    sentencepiece \
    psutil \
    filelock

# Install xformers compatible with torch 2.7
RUN pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu124

# Clone SGLang repository
# 2nd SHA occurrence: git checkout (REQUIRED)
RUN git clone https://github.com/sgl-project/sglang.git \
    && cd sglang \
    && git checkout da47621ccc4f8e8381f3249257489d5fe32aff1b

# Verify commit SHA and write to file
# 3rd SHA occurrence: verification (REQUIRED)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="da47621ccc4f8e8381f3249257489d5fe32aff1b" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch. Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.7

# Build flashinfer from source since no wheel available for 0.2.6.post1
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout v0.2.6.post1 \
    && cd python \
    && pip install ninja \
    && export TORCH_CUDA_ARCH_LIST="9.0" \
    && export MAX_JOBS=96 \
    && pip install --no-build-isolation . \
    && cd / \
    && rm -rf /tmp/flashinfer

# Install SGLang runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    transformers \
    uvicorn \
    uvloop \
    xgrammar==0.1.19

# Install additional SGLang srt dependencies
RUN pip install -c /opt/constraints.txt \
    cuda-python \
    einops

# Install SGLang in editable mode with --no-deps
WORKDIR /sgl-workspace/sglang/python
RUN pip install -e . --no-deps

# Install core SGLang dependencies
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Replace Triton with nightly version for compatibility
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Final sanity checks
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Verify commit SHA file exists and is correct
RUN test -f /opt/sglang_commit.txt && \
    COMMIT=$(cat /opt/sglang_commit.txt) && \
    test -n "$COMMIT" || exit 1 && \
    echo "Commit SHA file verified at /opt/sglang_commit.txt"

WORKDIR /sgl-workspace/sglang

ENV DEBIAN_FRONTEND=interactive

# Set entrypoint for SGLang server
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]