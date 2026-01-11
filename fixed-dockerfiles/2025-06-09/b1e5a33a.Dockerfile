# Fixed Dockerfile for SGLang commit b1e5a33ae337d20e35e966b8d82a02a913d32689
# Date: 2025-06-09
# Using torch 2.6.0 which requires Ubuntu 22.04 base image

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# HARDCODE #1: Set commit SHA as environment variable
ENV SGLANG_COMMIT=b1e5a33ae337d20e35e966b8d82a02a913d32689

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git curl wget vim \
    build-essential cmake ninja-build \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.6.0 and torchvision 0.21.0 (from pyproject.toml)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from PyPI for June 2025 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-06-09 era
# outlines 0.0.44 requires pydantic >=2.0 (discovered via metadata extraction)
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.0.3
transformers==4.52.3
huggingface_hub==0.23.2
numpy==1.26.4
scipy==1.13.1
pillow==10.3.0
requests==2.32.3
tqdm==4.66.4
aiohttp==3.9.5
msgspec==0.18.6
orjson==3.10.3
packaging==24.1
psutil==5.9.8
prometheus-client==0.20.0
pynvml==11.5.0
python-multipart==0.0.9
uvloop==0.19.0
einops==0.8.0
interegular==0.3.3
jinja2==3.1.4
lark==1.1.9
nest-asyncio==1.6.0
cloudpickle==3.0.0
diskcache==5.6.3
numba==0.60.0
referencing==0.35.1
jsonschema==4.22.0
datasets==2.19.2
pycountry==24.6.1
pyairports==2.1.1
compressed-tensors==0.4.0
partial-json-parser==0.2.1.1
setproctitle==1.3.3
blobfile==3.0.0
EOF

# Install vLLM 0.9.0 without dependencies (appropriate for June 9, 2025)
RUN pip install vllm==0.9.0 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    accelerate \
    aioprometheus \
    blake3 \
    click \
    cmake \
    cuda-python \
    dataclasses-json \
    diskcache \
    fastapi \
    filelock \
    gguf \
    importlib-metadata \
    jinja2 \
    jsonschema \
    lark \
    lm-format-enforcer \
    msgspec \
    nest-asyncio \
    ninja \
    numpy \
    nvidia-ml-py \
    openai \
    outlines \
    packaging \
    pillow \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    protobuf \
    psutil \
    py-cpuinfo \
    pydantic \
    pyzmq \
    ray \
    regex \
    requests \
    safetensors \
    scipy \
    sentencepiece \
    tiktoken \
    tokenizers \
    torch \
    torchvision \
    tqdm \
    transformers \
    triton \
    typing_extensions \
    uvicorn \
    uvloop \
    watchfiles \
    xformers

# Install sgl-kernel 0.1.6.post1 (available on PyPI)
RUN pip install sgl-kernel==0.1.6.post1

# Build flashinfer from source (no wheels available for cu124/torch2.6)
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.5 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install -e . --no-deps && \
    cd ../.. && rm -rf flashinfer

# HARDCODE #2: Clone SGLang at specific commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout b1e5a33ae337d20e35e966b8d82a02a913d32689

# HARDCODE #3: Verify commit SHA and write to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b1e5a33ae337d20e35e966b8d82a02a913d32689" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.11 \
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
    torchao==0.7.0 \
    transformers==4.52.3 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19

# Install SGLang in editable mode without dependencies
WORKDIR /sgl-workspace/sglang/python
RUN TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install -e . --no-deps

# Install remaining SGLang dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    cuda-python \
    outlines \
    einops

# Final cleanup
RUN pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace

# Set environment for runtime
ENV DEBIAN_FRONTEND=interactive
ENV CUDA_VISIBLE_DEVICES=0

# Entrypoint
CMD ["/bin/bash"]