# Fixed Dockerfile for SGLang commit efb099cd (2024-10-21)
# Base image for torch 2.4.x + CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip \
    && cd .. && rm -rf Python-3.10.14*

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 (required by vLLM 0.6.3.post1)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.6.3.post1 without dependencies to control versions
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies manually with correct versions
RUN pip3 install \
    nvidia-ml-py \
    torchvision==0.19 \
    xformers==0.0.27.post2 \
    psutil \
    sentencepiece \
    numpy \
    transformers \
    fastapi \
    aiohttp \
    openai \
    uvicorn \
    pydantic \
    pillow \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    pynvml \
    ray>=2.9 \
    pyzmq \
    msgpack \
    msgpack-numpy \
    packaging \
    py-cpuinfo \
    typing-extensions>=4.10 \
    filelock>=3.10.4 \
    partial-json-parser \
    lm-eval==0.4.4

# Build flashinfer from source (no wheels available for our combo)
RUN pip3 install ninja packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.1.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip3 install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=efb099cdee90b9ad332fcda96d89dd91ddebe072

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded in command)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout efb099cdee90b9ad332fcda96d89dd91ddebe072

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="efb099cdee90b9ad332fcda96d89dd91ddebe072" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install additional dependencies that might be needed
RUN pip3 install datamodel_code_generator

# Install SGLang from checked-out source
# The pyproject.toml is in the python/ subdirectory for this era of commits
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace triton with triton-nightly for better compatibility
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Clear pip cache
RUN pip3 cache purge

WORKDIR /sgl-workspace/sglang

# Set interactive frontend back
ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]