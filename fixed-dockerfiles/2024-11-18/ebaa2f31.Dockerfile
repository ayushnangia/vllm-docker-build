# Fixed Dockerfile for SGLang commit from November 18, 2024
# Date: 2024-11-18
# SGLang version: 0.3.5.post2
# vLLM: 0.6.3.post1 (requires torch 2.4.0)
# Target: H100 GPU
# Discovered versions via PyPI for November 18, 2024

# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# HARDCODE 1/3: ENV with commit SHA
ENV SGLANG_COMMIT=ebaa2f31996e80e4128b832d70f29f288b59944e
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# System dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        sudo \
        libibverbs-dev \
        ninja-build \
        cmake \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

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
    cd .. && \
    rm -rf Python-3.10.14*

# Upgrade pip and essential packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 with CUDA 12.1 (exact version for vLLM 0.6.3.post1)
RUN pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with versions discovered from PyPI for November 18, 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Discovered from PyPI for November 18, 2024
fastapi==0.115.5
uvicorn==0.32.0
pydantic==2.9.2
typing_extensions==4.12.2
outlines==0.0.46
pyzmq==26.2.0
aiohttp==3.10.10
pillow==10.4.0
protobuf==5.28.3
prometheus-client==0.21.0
prometheus-fastapi-instrumentator==7.0.0
tiktoken==0.8.0
lm-format-enforcer==0.10.6
filelock==3.16.1
partial-json-parser==0.2.1.1
msgspec==0.18.6
gguf==0.10.0
importlib_metadata==8.5.0
mistral_common==1.4.4
pyyaml==6.0.2
einops==0.8.0
compressed-tensors==0.6.0
orjson==3.10.11
huggingface_hub==0.26.2
transformers==4.46.2
tokenizers==0.20.3
ray==2.38.0
nvidia-ml-py==12.560.30
xformers==0.0.27.post2
EOF

# Install vLLM 0.6.3.post1 with --no-deps first
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
    "fastapi>=0.107.0,!=0.113.*,!=0.114.0" \
    aiohttp \
    "openai>=1.40.0" \
    "uvicorn[standard]" \
    "pydantic>=2.9" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    partial-json-parser \
    pyzmq \
    msgspec \
    "gguf==0.10.0" \
    importlib_metadata \
    "mistral_common[opencv]>=1.4.4" \
    pyyaml \
    "six>=1.16.0" \
    "setuptools>=74.1.1" \
    einops \
    "compressed-tensors==0.6.0"

# Install CUDA-specific vLLM dependencies
RUN pip3 install -c /opt/constraints.txt \
    "ray>=2.9" \
    nvidia-ml-py \
    "xformers==0.0.27.post2"

# Install flashinfer for torch 2.4 + CUDA 12.1
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ || \
    pip3 install flashinfer-python

# Install sgl-kernel (version from the era)
RUN pip3 install sgl-kernel==0.3.5.post2 || pip3 install sgl-kernel

# Install triton
RUN pip3 install triton>=2.0.0

# Working directory
WORKDIR /sgl-workspace

# HARDCODE 2/3: Clone and checkout exact commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout ebaa2f31996e80e4128b832d70f29f288b59944e

# HARDCODE 3/3: Verify commit SHA
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ebaa2f31996e80e4128b832d70f29f288b59944e" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps first
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    IPython \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    "pyzmq>=25.1.2" \
    "outlines>=0.0.44,<0.1.0" \
    modelscope

# Additional packages
RUN pip3 install datamodel_code_generator

# Try to upgrade triton if needed (sometimes helps with compatibility)
RUN pip3 uninstall -y triton triton-nightly 2>/dev/null || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly || \
    pip3 install triton>=2.0.0

# Final verification
RUN python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import vllm; print('vLLM import successful')" && \
    python3 -c "import outlines; print('Outlines import successful')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import typing_extensions; print(f'typing_extensions version: {typing_extensions.__version__}')"

# Clean pip cache
RUN python3 -m pip cache purge

# Reset to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

# Set working directory for runtime
WORKDIR /sgl-workspace/sglang

# Print commit info
RUN echo "Built SGLang at commit: $(cat /opt/sglang_commit.txt)"

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]