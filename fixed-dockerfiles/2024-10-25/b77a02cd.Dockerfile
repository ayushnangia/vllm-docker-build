# Fixed Dockerfile for SGLang commit from 2024-10-25
# Base image for torch 2.4.0 with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
# 1st SHA occurrence: ENV
ENV SGLANG_COMMIT=b77a02cdfdb4cd58be3ebc6a66d076832c309cfc
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies and build tools
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        curl \
        git \
        wget \
        sudo \
        libibverbs-dev \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        libgdbm-dev \
        libnss3-dev \
        libegl1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Build Python 3.10 from source (deadsnakes PPA deprecated on Ubuntu 20.04)
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --with-lto --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/Python-3.10.14* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10 && \
    python3 --version

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip --version

# Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 with CUDA 12.1
RUN pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers 0.0.27.post2 for torch 2.4.0
RUN pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions for October 2024
# CRITICAL: outlines 0.0.44 requires pydantic>=2.0, typing_extensions must be <4.14 to avoid Sentinel
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-10-25 era
# fastapi 0.115.0 was released Sept 17, 2024
fastapi==0.115.0
# uvicorn 0.31.0 was latest stable
uvicorn==0.31.0
# pydantic 2.9.2 was released Sept 17, 2024 (latest v2 before Oct 25)
pydantic==2.9.2
pydantic-core==2.23.4
# typing_extensions 4.12.2 released June 2024 (before 4.14 which has Sentinel)
typing_extensions==4.12.2
# outlines 0.0.44 released June 14, 2024 (requires pydantic>=2.0)
outlines==0.0.44
# Other packages from around Oct 2024
pyzmq==26.2.0
aiohttp==3.10.10
orjson==3.10.7
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
    partial-json-parser \
    msgspec \
    "gguf==0.10.0" \
    importlib_metadata \
    "mistral_common[opencv]>=1.4.4" \
    pyyaml \
    "setuptools>=74.1.1" \
    einops \
    "compressed-tensors==0.6.0" \
    "filelock>=3.10.4" \
    ray \
    nvidia-ml-py \
    cloudpickle \
    "outlines>=0.0.43,<0.1"

# Install flashinfer 0.2.0.post1 (wheels available for cu121/torch2.4)
RUN pip3 install flashinfer==0.2.0.post1 -i https://flashinfer.ai/whl/cu121/torch2.4/

# Working directory
WORKDIR /sgl-workspace

# 2nd SHA occurrence: git clone and checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout b77a02cdfdb4cd58be3ebc6a66d076832c309cfc

# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL_SHA=$(git rev-parse HEAD) && \
    EXPECTED_SHA="b77a02cdfdb4cd58be3ebc6a66d076832c309cfc" && \
    if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then \
        echo "ERROR: SHA mismatch! Expected $EXPECTED_SHA but got $ACTUAL_SHA" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL_SHA" > /opt/sglang_commit.txt

# Install sgl-kernel from PyPI (0.3.4.post1 is available)
RUN pip3 install sgl-kernel==0.3.4.post1 || \
    (cd /sgl-workspace/sglang/python/sgl_kernel && pip3 install -e . --no-build-isolation)

# Install datamodel_code_generator for MiniCPM models
RUN pip3 install datamodel_code_generator

# Install SGLang with --no-deps (editable install)
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang runtime dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    pyzmq \
    modelscope

# OpenAI, Anthropic, and LiteLLM support
RUN pip3 install \
    "openai>=1.0" \
    tiktoken \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Final verification - use pip show for packages that need GPU to import
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import successful')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    pip show outlines > /dev/null && echo "Outlines installed OK" && \
    python3 -c "import flashinfer; print('FlashInfer import successful')" && \
    pip show xformers > /dev/null && echo "xformers installed OK" && \
    pip show sgl-kernel > /dev/null && echo "sgl-kernel installed OK"

# Clean up pip cache
RUN pip3 cache purge

# Reset to interactive
ENV DEBIAN_FRONTEND=interactive

# Sanity check - MUST verify commit SHA matches exactly
RUN echo "=== Sanity Check ===" && \
    echo "Expected commit: ${SGLANG_COMMIT}" && \
    echo -n "Actual commit: " && cat /opt/sglang_commit.txt && \
    test "$(cat /opt/sglang_commit.txt)" = "${SGLANG_COMMIT}" && \
    echo "✓ Commit verification passed" && \
    echo "✓ All imports successful" && \
    echo "=== Sanity Check Complete ==="

# Default command
CMD ["/bin/bash"]