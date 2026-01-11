# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
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
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from PyPI wheel (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.5.5 as specified in pyproject.toml
RUN pip3 install vllm==0.5.5

# HARDCODE the commit SHA (1/3)
ENV SGLANG_COMMIT=c98e84c21e4313d7d307425ca43e61753a53a9f7

# Clone SGLang and checkout EXACT commit (2/3 - hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout c98e84c21e4313d7d307425ca43e61753a53a9f7

# VERIFY commit - compare against HARDCODED value (3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="c98e84c21e4313d7d307425ca43e61753a53a9f7" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
WORKDIR /sgl-workspace/sglang
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i '/^\s*$/d' /sgl-workspace/sglang/python/pyproject.toml

# Install additional dependencies for MiniCPM models
RUN pip3 install datamodel_code_generator

# Install SGLang from source with all dependencies
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace Triton with nightly version for better compatibility
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')"

# Clean pip cache
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]