# Dockerfile for SGLang (2024-10-17)
# Based on pyproject.toml: sglang 0.3.3.post1, vllm==0.5.5
# Date suggests torch 2.4.x era with CUDA 12.1

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# HARDCODED commit SHA - this Dockerfile is specific to this commit
ENV SGLANG_COMMIT=b170930534acbb9c1619a3c83670a839ceee763a

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
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

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and install build tools
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 support (compatible with vLLM 0.5.5)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 as specified in pyproject.toml
RUN pip3 install vllm==0.5.5

# Install flashinfer for torch 2.4 + CUDA 12.1 (try wheel first)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ || \
    (echo "Flashinfer wheel not found, building from source..." && \
     pip3 install ninja numpy packaging && \
     git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
     cd /tmp/flashinfer && \
     git checkout v0.1.6 && \
     cd python && \
     TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
     rm -rf /tmp/flashinfer)

# Clone SGLang at EXACT commit (hardcoded SHA, not variable)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout b170930534acbb9c1619a3c83670a839ceee763a

# VERIFY the checkout matches expected commit (hardcoded comparison)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b170930534acbb9c1619a3c83670a839ceee763a" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove flashinfer if it was pre-installed
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install additional dependencies that might be needed
RUN pip3 install datamodel_code_generator transformers accelerate

# Install SGLang from checked-out source
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace Triton with nightly version for better compatibility
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')"

# Clean up pip cache
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]