# Fixed Dockerfile for SGLang commit 8f8f96a6217ea737c94e7429e480196319594459
# Date: 2024-10-23
# SGLang version: 0.3.4.post1
# vLLM: 0.6.3.post1
# torch: 2.5.1 (better compatibility with vLLM 0.6.x)

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo libibverbs-dev \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    cmake ninja-build \
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

# Upgrade pip and essential packages
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.5.1 with CUDA 12.1 BEFORE other packages
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install numpy and packaging first (needed for builds)
RUN pip3 install numpy packaging

# Build flashinfer from source (v0.1.6 for better compatibility)
RUN pip3 install ninja pybind11 && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.1.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=$(nproc) pip3 install --no-build-isolation . && \
    cd / && rm -rf /tmp/flashinfer

# Install xformers compatible with torch 2.5.x
RUN pip3 install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.6.3.post1 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies manually (excluding torch and xformers which we already installed)
RUN pip3 install \
    "nvidia-ml-py" \
    "psutil" \
    "ray>=2.9" \
    "sentencepiece" \
    "transformers>=4.45" \
    "fastapi" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "prometheus-client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.9" \
    "outlines>=0.0.43" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "pyzmq" \
    "msgspec" \
    "gguf==0.9.1"

# Install sgl-kernel from PyPI (version available)
RUN pip3 install sgl-kernel==0.3.4.post1

# Install other SGLang dependencies
RUN pip3 install \
    "aiohttp" \
    "decord" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "orjson" \
    "pillow" \
    "python-multipart" \
    "torchao" \
    "uvloop" \
    "modelscope" \
    "openai>=1.0" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0" \
    "datamodel_code_generator" \
    "requests" \
    "tqdm"

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=8f8f96a6217ea737c94e7429e480196319594459

# Clone SGLang repository and checkout EXACT commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 8f8f96a6217ea737c94e7429e480196319594459

# VERIFY the checkout - compare against HARDCODED expected value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="8f8f96a6217ea737c94e7429e480196319594459" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
WORKDIR /sgl-workspace/sglang/python
RUN sed -i 's/"flashinfer[^"]*",*//g' pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' pyproject.toml

# Install SGLang from checked-out source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install triton-nightly for better performance
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify all critical imports work
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')"

# Clean pip cache
RUN pip3 cache purge

# Set working directory
WORKDIR /sgl-workspace/sglang

# Set entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]