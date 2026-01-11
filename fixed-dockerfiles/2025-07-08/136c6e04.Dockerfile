# Base image for torch 2.7.1 with CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=96

# 1st hardcoded commit SHA occurrence: ENV variable
ENV SGLANG_COMMIT=136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git curl wget build-essential cmake ninja-build \
    ccache vim \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.7.1 (as specified in pyproject.toml)
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from PyPI for July 2025
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-07-08 era
fastapi==0.116.0
uvicorn==0.35.0
pydantic==2.11.5
pydantic-core==2.27.1
typing_extensions==4.13.2
outlines==0.1.11
pyzmq==27.0.0
prometheus-client==0.20.0
EOF

# Install vLLM 0.9.2 (released July 8, 2025)
RUN pip install vllm==0.9.2 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    ray \
    sentencepiece \
    numpy \
    transformers \
    xformers \
    fastapi \
    uvicorn \
    pydantic \
    aioprometheus \
    pynvml \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing_extensions \
    filelock \
    pyzmq \
    msgspec \
    gguf \
    compressed-tensors \
    lark \
    scipy \
    pillow \
    packaging

# Install torchao (specified in pyproject.toml)
RUN pip install torchao==0.9.0

# Build and install flashinfer from source (no wheels for torch 2.7)
WORKDIR /tmp
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.7 && \
    cd python && \
    pip install . && \
    cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel from PyPI (version 0.2.4 is available)
RUN pip install sgl-kernel==0.2.4

# 2nd hardcoded commit SHA occurrence: git checkout
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7

# 3rd hardcoded commit SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "Error: Commit mismatch. Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps first
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    blobfile==3.0.0 \
    build \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.13 \
    modelscope \
    msgspec \
    ninja \
    orjson \
    outlines==0.1.11 \
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
    transformers==4.53.0 \
    timm==1.0.16 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19 \
    cuda-python \
    einops

# Sanity check: Verify SGLang installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import flashinfer; print('FlashInfer imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Verify the commit SHA file exists
RUN test -f /opt/sglang_commit.txt && \
    COMMIT=$(cat /opt/sglang_commit.txt) && \
    echo "Commit in container: $COMMIT"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default command
CMD ["/bin/bash"]