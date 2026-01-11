# SGLang Dockerfile for commit 83452dbb4a19c6a2461e972eb2b64a2df9a466b8
# Build date: 2025-01-19
# Based on discovered dependencies via WebFetch/WebSearch

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=83452dbb4a19c6a2461e972eb2b64a2df9a466b8
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies and Python 3.10
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        software-properties-common \
        curl \
        git \
        wget \
        sudo \
        build-essential \
        ninja-build \
        libibverbs-dev \
        ccache \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Build Python 3.10 from source (Ubuntu 20.04 deadsnakes PPA is broken)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 \
    && cd .. && rm -rf Python-3.10.14*

WORKDIR /sgl-workspace

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Install PyTorch 2.4.0 with CUDA 12.1
RUN python3 -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-01-19 era
# outlines 0.0.44 requires pydantic >= 2.0 (verified via wheel METADATA)
# typing_extensions 4.14.0+ has Sentinel (breaks pydantic-core), using 4.12.2
fastapi==0.115.7
uvicorn==0.34.0
pydantic==2.10.5
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
EOF

# Install vllm 0.6.3.post1 with --no-deps first
RUN python3 -m pip install vllm==0.6.3.post1 --no-deps

# Install vllm dependencies from its requirements-common.txt (discovered via repo exploration)
RUN python3 -m pip install -c /opt/constraints.txt \
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
    "einops" \
    "compressed-tensors==0.6.0"

# Install vllm CUDA-specific dependencies (from requirements-cuda.txt)
RUN python3 -m pip install \
    "ray>=2.9" \
    nvidia-ml-py \
    "xformers==0.0.27.post2"

# Clone SGLang at the specific commit (2nd SHA occurrence)
RUN git clone https://github.com/sgl-project/sglang.git sglang \
    && cd sglang \
    && git checkout 83452dbb4a19c6a2461e972eb2b64a2df9a466b8

# Verify the commit SHA and write to file (3rd SHA occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="83452dbb4a19c6a2461e972eb2b64a2df9a466b8" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Build sgl-kernel from source (not available on PyPI until April 2025)
RUN cd /sgl-workspace/sglang/python && \
    if [ -d "sglang/srt/layers/kernels" ]; then \
        cd sglang/srt/layers/kernels && \
        python3 setup.py bdist_wheel && \
        pip install dist/*.whl; \
    else \
        echo "Warning: sgl-kernel directory not found, trying to build from repo" && \
        git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel && \
        cd /tmp/sgl-kernel && \
        TORCH_CUDA_ARCH_LIST="9.0" pip3 install . && \
        rm -rf /tmp/sgl-kernel; \
    fi

# Install flashinfer 0.1.6 with CUDA 12.1 and torch 2.4 (verified available via WebFetch)
RUN python3 -m pip install flashinfer==0.1.6 \
    --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang && \
    python3 -m pip install -e python/ --no-deps

# Install SGLang runtime_common dependencies from pyproject.toml
RUN python3 -m pip install -c /opt/constraints.txt \
    aiohttp \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    orjson \
    packaging \
    "prometheus-client>=0.20.0" \
    psutil \
    python-multipart \
    uvloop \
    "xgrammar>=0.1.6"

# Install SGLang srt dependencies (that aren't already installed)
RUN python3 -m pip install \
    cuda-python \
    "torchao>=0.7.0"

# Install additional SGLang project dependencies from pyproject.toml
RUN python3 -m pip install \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# For MiniCPM models (from original Dockerfile)
RUN python3 -m pip install datamodel_code_generator

# Final verification - check imports and commit SHA
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" && \
    python3 -c "import typing_extensions; print(f'typing_extensions version: {typing_extensions.__version__}')" && \
    test "$(cat /opt/sglang_commit.txt)" = "83452dbb4a19c6a2461e972eb2b64a2df9a466b8" || exit 1

# Clean up pip cache
RUN python3 -m pip cache purge

# Reset to interactive mode
ENV DEBIAN_FRONTEND=interactive

# Entry point
WORKDIR /sgl-workspace
CMD ["/bin/bash"]