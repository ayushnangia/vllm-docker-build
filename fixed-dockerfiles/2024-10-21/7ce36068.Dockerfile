# Dockerfile for SGLang commit 7ce36068914503c3a53ad7be23ab29831fb8aa63
# Date: 2024-10-21
# vLLM: 0.6.3.post1 (requires torch 2.4.0)

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev \
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
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from the custom index (flashinfer_python is the new name)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.6.3.post1
RUN pip3 install vllm==0.6.3.post1

# Install additional dependencies for vLLM/SGLang compatibility
RUN pip3 install transformers>=4.45.2 tokenizers>=0.19.1

# HARDCODE the commit SHA (don't use ARG to avoid forgotten --build-arg issues)
ENV SGLANG_COMMIT=7ce36068914503c3a53ad7be23ab29831fb8aa63

# Clone SGLang and checkout EXACT commit
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 7ce36068914503c3a53ad7be23ab29831fb8aa63

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="7ce36068914503c3a53ad7be23ab29831fb8aa63" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from checked-out source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install datamodel_code_generator for MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# Clean pip cache
RUN pip3 cache purge

# Verify installation
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Set interactive frontend back
ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
