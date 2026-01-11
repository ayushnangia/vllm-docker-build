# Fixed Dockerfile for SGLang commit 93470a14116a60fe5dd43f0599206e8ccabdc211
# Date: 2025-04-07
# torch==2.5.1, flashinfer_python==0.2.3, sgl-kernel==0.0.8

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev \
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
    && cd .. && rm -rf Python-3.10.14*

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.5.1 with CUDA 12.1
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer_python 0.2.3 from wheel (available for torch 2.5 + CUDA 12.1)
RUN pip3 install flashinfer-python==0.2.3 -i https://flashinfer.ai/whl/cu121/torch2.5/flashinfer-python/

# Build sgl-kernel from source (version 0.0.8 not on PyPI)
RUN pip3 install ninja numpy packaging \
    && git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel \
    && cd /tmp/sgl-kernel \
    && git checkout v0.0.8 \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . \
    && rm -rf /tmp/sgl-kernel

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=93470a14116a60fe5dd43f0599206e8ccabdc211

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 93470a14116a60fe5dd43f0599206e8ccabdc211

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="93470a14116a60fe5dd43f0599206e8ccabdc211" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i '/^[[:space:]]*$/d' python/pyproject.toml && \
    sed -i 's/,,/,/g' python/pyproject.toml && \
    sed -i 's/,]/]/g' python/pyproject.toml

# Install additional dependencies
RUN pip3 install transformers==4.51.0

# Install SGLang from source
WORKDIR /sgl-workspace/sglang
RUN cd python && pip3 install -e ".[srt]"

# Install Triton nightly (often needed for modern SGLang)
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('SGL-kernel import OK')"

# Final environment setup
ENV DEBIAN_FRONTEND=interactive
WORKDIR /sgl-workspace/sglang

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]