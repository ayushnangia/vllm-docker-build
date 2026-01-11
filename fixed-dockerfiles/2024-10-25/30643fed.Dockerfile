# SGLang Dockerfile for commit 30643fed7f92be32540dfcdf9e4310e477ce0f6d
# Date: 2024-10-25
# SGLang version: 0.3.4.post2
# vLLM: 0.6.3.post1, torch: 2.4.1

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

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and install build tools
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.1 with CUDA 12.1 BEFORE sglang
RUN pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from PyPI wheels (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.6.3.post1 as required by pyproject.toml
RUN pip3 install vllm==0.6.3.post1

# For openbmb/MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=30643fed7f92be32540dfcdf9e4310e477ce0f6d

# Clone SGLang and checkout EXACT commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 30643fed7f92be32540dfcdf9e4310e477ce0f6d

# VERIFY commit - compare against HARDCODED value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="30643fed7f92be32540dfcdf9e4310e477ce0f6d" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed flashinfer
# (flashinfer not in this pyproject.toml, but adding for safety)
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml || true && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml || true

# Install SGLang from source in editable mode
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install triton-nightly as recommended for better performance
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100 GPUs
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Clean pip cache
RUN pip3 cache purge

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists with content:" && \
    cat /opt/sglang_commit.txt

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]