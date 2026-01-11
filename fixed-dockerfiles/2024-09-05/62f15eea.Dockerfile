# Fixed Dockerfile for SGLang commit 62f15eea5a0b4266cdae965d0337fd33f6673736
# Date: 2024-09-05
# Base: nvidia/cuda:12.1.1-devel-ubuntu20.04 (for torch 2.4.x compatibility)

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    sudo \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
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
    && cd .. && rm -rf Python-3.10.14*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# HARDCODE #1: Set the commit SHA as environment variable
ENV SGLANG_COMMIT=62f15eea5a0b4266cdae965d0337fd33f6673736

# Pre-install torch 2.4.0 with CUDA 12.1 support
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vllm==0.5.5 as specified in pyproject.toml
RUN pip3 install vllm==0.5.5

# Install flashinfer from wheel (available for cu121/torch2.4)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install other dependencies that might be needed
RUN pip3 install packaging ninja numpy

# HARDCODE #2: Clone SGLang and checkout exact commit
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 62f15eea5a0b4266cdae965d0337fd33f6673736

# HARDCODE #3: Verify the commit matches exactly
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="62f15eea5a0b4266cdae965d0337fd33f6673736" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source with all dependencies
WORKDIR /sgl-workspace/sglang
RUN cd python && pip3 install -e ".[all]"

# Install triton-nightly to avoid conflicts
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Clean pip cache
RUN pip3 cache purge

# Verify SGLang installation
RUN python3 -c "import sglang; print('SGLang import OK')"

# Verify flashinfer installation
RUN python3 -c "import flashinfer; print('Flashinfer import OK')"

# Verify vllm installation
RUN python3 -c "import vllm; print('vLLM import OK')"

# Set working directory
WORKDIR /sgl-workspace

# Set interactive frontend back
ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]