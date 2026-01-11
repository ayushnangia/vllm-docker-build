# Fixed Dockerfile for SGLang commit (2025-04-07)
# Date: 2025-04-07
# torch: 2.5.1, flashinfer_python: 0.2.3 (build from source), sgl-kernel: 0.0.8 (build from source)

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

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
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip \
    && cd .. && rm -rf Python-3.10.14*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.5.1 with CUDA 12.1 index
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Build flashinfer from source (version 0.2.3 not available as wheel for torch 2.5)
RUN pip install ninja numpy packaging \
    && git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout v0.2.3 \
    && cd python \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . \
    && cd / && rm -rf /tmp/flashinfer

# Build sgl-kernel from source (version 0.0.8 not on PyPI)
RUN git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel \
    && cd /tmp/sgl-kernel \
    && git checkout v0.0.8 \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . \
    && cd / && rm -rf /tmp/sgl-kernel

# HARDCODE commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=db452760e5b2378efd06b1ceb9385d2eeb6d217c

# Clone SGLang and checkout EXACT commit (occurrence 2/3 - hardcoded in checkout command)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout db452760e5b2378efd06b1ceb9385d2eeb6d217c

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="db452760e5b2378efd06b1ceb9385d2eeb6d217c" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml

# Install SGLang from checked-out source
WORKDIR /sgl-workspace/sglang
RUN cd python && pip install -e ".[all]"

# Install any additional dependencies that might be needed
RUN pip install transformers==4.51.0

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')"

# Final verification of commit
RUN cat /opt/sglang_commit.txt

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]