# SGLang Dockerfile for commit db452760e5b2378efd06b1ceb9385d2eeb6d217c (2025-04-07)
# Base image for torch 2.5.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=db452760e5b2378efd06b1ceb9385d2eeb6d217c
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update -y && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    ca-certificates \
    ccache \
    libjpeg-dev \
    libpng-dev \
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

# Verify Python installation and upgrade pip
RUN python3 --version && \
    pip3 --version && \
    pip3 install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 for CUDA 12.1
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI (2025-04-07 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-04-07 era
fastapi==0.115.12
uvicorn==0.34.0
pydantic==2.11.3
typing_extensions==4.13.1
outlines==0.0.44
pyzmq==26.4.0
transformers==4.51.0
numpy<2.0
EOF

# Install vLLM 0.8.3 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.8.3 --no-deps

# Install vLLM dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    accelerate \
    aiohttp \
    cloudpickle \
    einops \
    filelock \
    huggingface-hub \
    importlib-metadata \
    jinja2 \
    jsonschema \
    lm-format-enforcer \
    msgspec \
    ninja \
    openai \
    packaging \
    pillow \
    prometheus_client \
    protobuf \
    psutil \
    py-cpuinfo \
    pydantic \
    pyzmq \
    ray \
    requests \
    safetensors \
    sentencepiece \
    tiktoken \
    tokenizers \
    tqdm \
    transformers \
    typing_extensions \
    uvicorn \
    uvloop \
    watchfiles \
    xformers

# Install flashinfer from prebuilt wheel
RUN pip3 install flashinfer-python==0.2.3 \
    -i https://flashinfer.ai/whl/cu121/torch2.5/flashinfer-python/

# Clone SGLang at specific commit (2nd hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout db452760e5b2378efd06b1ceb9385d2eeb6d217c

# Verify commit SHA and write to file (3rd hardcoded SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="db452760e5b2378efd06b1ceb9385d2eeb6d217c" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED, got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Build sgl-kernel from source (0.0.8 not available on PyPI)
RUN git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel && \
    cd /tmp/sgl-kernel && \
    git checkout v0.0.8 && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    rm -rf /tmp/sgl-kernel

# Install SGLang with --no-deps to avoid dependency conflicts
WORKDIR /sgl-workspace/sglang
RUN pip3 install --no-deps -e python/

# Install SGLang runtime dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    compressed-tensors \
    datasets \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance \
    modelscope \
    ninja \
    orjson \
    packaging \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    soundfile==0.13.1 \
    torchao \
    transformers \
    uvicorn \
    uvloop \
    xgrammar==0.1.17 \
    partial_json_parser \
    einops \
    cuda-python \
    outlines

# Sanity check: verify imports
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import successful')" && \
    python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import flashinfer; print('FlashInfer import successful')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import successful')" && \
    python3 -c "import outlines; print('Outlines import successful')"

# Final verification of commit SHA
RUN test "$(cat /opt/sglang_commit.txt)" = "db452760e5b2378efd06b1ceb9385d2eeb6d217c" || \
    (echo "ERROR: Commit SHA mismatch!" && exit 1)

# Set working directory
WORKDIR /workspace

# Set environment back to interactive
ENV DEBIAN_FRONTEND=interactive

# Default command
CMD ["/bin/bash"]