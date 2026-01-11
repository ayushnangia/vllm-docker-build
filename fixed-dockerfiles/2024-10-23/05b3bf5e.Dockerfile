# SGLang Dockerfile for commit 05b3bf5e (2024-10-23)
# Full commit SHA: 05b3bf5e8e4751cf51510198ae2e864c4b11ac2f
# Base image for torch 2.4.x
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# 1st SHA occurrence - ENV
ENV SGLANG_COMMIT=05b3bf5e8e4751cf51510198ae2e864c4b11ac2f

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    sudo \
    libibverbs-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
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
RUN python3 --version && python3 -m pip --version

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install torch 2.4.0
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-10-23 era
# pydantic 2.9.2 - Released Sep 17, 2024 (latest before Oct 23)
pydantic==2.9.2
# typing_extensions 4.12.2 - Released Jun 7, 2024
typing_extensions==4.12.2
# fastapi 0.112.4 - Compatible version (not 0.113.* or 0.114.0)
fastapi==0.112.4
# uvicorn 0.30.6 - Latest 0.30.x series for stability
uvicorn==0.30.6
# outlines 0.0.44 - Required by SGLang pyproject.toml (requires pydantic>=2.0)
outlines==0.0.44
# pyzmq 26.2.0 - Released Jun 25, 2024
pyzmq==26.2.0
EOF

# Install vLLM 0.6.3.post1 with --no-deps
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with constraints
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
    aiohttp \
    "openai>=1.40.0" \
    "uvicorn[standard]" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "filelock>=3.10.4" \
    partial-json-parser \
    msgspec \
    "gguf==0.10.0" \
    importlib_metadata \
    "mistral_common[opencv]>=1.4.4" \
    pyyaml \
    nvidia-ml-py \
    "ray>=2.9" \
    xformers==0.0.27.post2 \
    torchvision==0.19.0

# Install additional dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    pydantic \
    fastapi \
    typing_extensions \
    outlines \
    pyzmq

# Install flashinfer from prebuilt wheels (version 0.1.6 was latest in Oct 2024)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone and checkout SGLang at specific commit
WORKDIR /sgl-workspace
# 2nd SHA occurrence - git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 05b3bf5e8e4751cf51510198ae2e864c4b11ac2f

# Verify commit SHA and write to file
# 3rd SHA occurrence - verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="05b3bf5e8e4751cf51510198ae2e864c4b11ac2f" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "Error: Commit mismatch. Expected $EXPECTED but got $ACTUAL" && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang runtime dependencies from pyproject.toml with constraints
RUN pip3 install -c /opt/constraints.txt \
    aiohttp \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    pillow \
    psutil \
    python-multipart \
    torchao \
    uvloop \
    modelscope

# Install datamodel_code_generator for MiniCPM models
RUN pip3 install datamodel_code_generator

# Install additional SGLang optional dependencies
RUN pip3 install \
    "openai>=1.0" \
    tiktoken \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Sanity check - verify everything imports correctly
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import fastapi; print('FastAPI imported successfully')" && \
    python3 -c "import pydantic; print('Pydantic imported successfully')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')"

# Verify versions match our constraints
RUN python3 -c "import importlib.metadata; print('pydantic:', importlib.metadata.version('pydantic'))" && \
    python3 -c "import importlib.metadata; print('fastapi:', importlib.metadata.version('fastapi'))" && \
    python3 -c "import importlib.metadata; print('typing_extensions:', importlib.metadata.version('typing_extensions'))" && \
    python3 -c "import importlib.metadata; print('outlines:', importlib.metadata.version('outlines'))" && \
    python3 -c "import importlib.metadata; print('vllm:', importlib.metadata.version('vllm'))" && \
    python3 -c "import torch; print(f'torch: {torch.__version__} with CUDA {torch.version.cuda}')"

# Clean pip cache
RUN pip3 cache purge

# Set environment back to interactive
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]