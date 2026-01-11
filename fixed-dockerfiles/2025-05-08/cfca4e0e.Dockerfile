# Fixed Dockerfile for SGLang commit cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a
# Date: 2025-05-08
# Dependencies from pyproject.toml:
# - torch==2.6.0, torchvision==0.21.0
# - flashinfer_python==0.2.5 (BUILD FROM SOURCE - no prebuilt wheel)
# - sgl-kernel==0.1.1 (available on PyPI)
# - outlines>=0.0.44,<=0.1.11 (requires pydantic>=2.0)
# - transformers==4.51.1, xgrammar==0.1.17

# Base image for torch 2.6.0 - requires CUDA 12.4+ and Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Build arguments and environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# HARDCODED COMMIT SHA - occurrence 1/3
ENV SGLANG_COMMIT=cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    libibverbs-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.6.0 and torchvision 0.21.0 with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from PyPI (May 2025 era)
# These versions were discovered via WebFetch from PyPI release histories
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-05-08 era
fastapi==0.111.0
uvicorn==0.34.2
pydantic==2.11.4
pydantic-core==2.30.1
typing_extensions==4.13.2
outlines==0.1.11
pyzmq==26.4.0
orjson==3.10.18
prometheus-client==0.22.0
uvloop==0.21.0
EOF

# Install vLLM 0.8.5.post1 with --no-deps to control versions
RUN pip install vllm==0.8.5.post1 --no-deps

# Install vLLM core dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    accelerate \
    aiohappyeyeballs \
    aiohttp \
    aiosignal \
    anytree \
    async-timeout \
    attrs \
    certifi \
    charset-normalizer \
    click \
    cloudpickle \
    cmake \
    compressed-tensors \
    diskcache \
    einops \
    exceptiongroup \
    filelock \
    frozenlist \
    fsspec \
    gguf \
    grpcio \
    h11 \
    httpcore \
    httptools \
    huggingface-hub \
    humanfriendly \
    idna \
    importlib-metadata \
    interegular \
    jinja2 \
    jsonschema \
    jsonschema-specifications \
    lark \
    markupsafe \
    mistral-common \
    mosaicml-streaming \
    mpmath \
    msgpack \
    msgspec \
    multidict \
    networkx \
    ninja \
    numpy \
    nvidia-ml-py \
    openai \
    packaging \
    pandas \
    partial-json-parser \
    pillow \
    prometheus-client \
    protobuf \
    psutil \
    py-cpuinfo \
    pydantic \
    pydantic-core \
    pynvml \
    python-dateutil \
    python-dotenv \
    python-multipart \
    pytz \
    pyyaml \
    pyzmq \
    ray \
    referencing \
    regex \
    requests \
    rpds-py \
    safetensors \
    scipy \
    sentencepiece \
    six \
    sniffio \
    starlette \
    sympy \
    tiktoken \
    tokenizers \
    tqdm \
    transformers==4.51.1 \
    triton \
    typing-extensions \
    tzdata \
    urllib3 \
    uvicorn \
    uvloop \
    watchfiles \
    websockets \
    xgrammar==0.1.17 \
    yarl \
    zstandard

# Install outlines 0.1.11 (requires pydantic>=2.0)
RUN pip install -c /opt/constraints.txt outlines==0.1.11

# Build flashinfer from source since 0.2.5 wheels aren't available
# (WebFetch confirmed no 0.2.5 wheels at flashinfer.ai/whl)
WORKDIR /tmp
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.5 && \
    cd python && \
    pip install -e . --no-build-isolation && \
    cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel 0.1.1 from PyPI (confirmed available via WebFetch)
RUN pip install sgl-kernel==0.1.1

# Clone SGLang at specific commit - HARDCODED SHA occurrence 2/3
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a

# Verify commit SHA and write to file - HARDCODED SHA occurrence 3/3
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="cfca4e0ed2cf4a97c2ee3b668f7115b59db0028a" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang Python package in editable mode with --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# Install SGLang runtime dependencies from pyproject.toml with constraints
RUN pip install -c /opt/constraints.txt \
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
    llguidance==0.7.11 \
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
    torchao==0.9.0 \
    transformers==4.51.1 \
    uvicorn \
    uvloop \
    xgrammar==0.1.17 \
    blobfile==3.0.0 \
    cuda-python \
    outlines \
    partial_json_parser \
    einops

# For openbmb/MiniCPM models
RUN pip install datamodel_code_generator

# Final sanity check - verify all key imports work
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import outlines; print('Outlines import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import importlib.metadata; print('Versions from PyPI discovery:'); \
                 print(f'  outlines: {importlib.metadata.version(\"outlines\")}'); \
                 print(f'  pydantic: {importlib.metadata.version(\"pydantic\")}'); \
                 print(f'  typing_extensions: {importlib.metadata.version(\"typing_extensions\")}'); \
                 print(f'  fastapi: {importlib.metadata.version(\"fastapi\")}'); \
                 print(f'  uvicorn: {importlib.metadata.version(\"uvicorn\")}')"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    COMMIT=$(cat /opt/sglang_commit.txt) && \
    echo "Docker image built for SGLang commit: $COMMIT"

# Set working directory
WORKDIR /sgl-workspace

ENV DEBIAN_FRONTEND=interactive