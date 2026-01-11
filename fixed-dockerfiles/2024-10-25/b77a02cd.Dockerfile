# Fixed Dockerfile for SGLang commit from 2024-10-25
# Date: 2024-10-25
# SGLang version: 0.3.4.post2
# vLLM version: 0.6.3.post1

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo libibverbs-dev \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
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

# Pre-install torch 2.4.1 with CUDA 12.1 index
RUN pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from wheel (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.6.3.post1 as specified in pyproject.toml
RUN pip3 install vllm==0.6.3.post1

# For openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator

# HARDCODE the commit SHA (exact 40-character value)
ENV SGLANG_COMMIT=b77a02cdfdb4cd58be3ebc6a66d076832c309cfc

# Clone SGLang and checkout EXACT commit (hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout b77a02cdfdb4cd58be3ebc6a66d076832c309cfc

# VERIFY commit - compare against HARDCODED value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b77a02cdfdb4cd58be3ebc6a66d076832c309cfc" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
WORKDIR /sgl-workspace/sglang
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source with all optional dependencies
# Using editable install to ensure proper source linkage
RUN cd /sgl-workspace/sglang && \
    pip3 install -e "python[all]"

# Replace triton with triton-nightly if needed (common for this era)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set torch CUDA arch list for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Clean pip cache
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]