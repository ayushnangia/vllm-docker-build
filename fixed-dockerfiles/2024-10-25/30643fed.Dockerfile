# SGLang Dockerfile for commit 30643fed
# Date: 2024-10-25
# SGLang version: 0.3.4.post2
# vLLM: 0.6.3.post1, torch: 2.4.0

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

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

# Pre-install torch 2.4.0 with CUDA 12.1 BEFORE sglang (vLLM requires exactly 2.4.0)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers 0.0.27.post2 (required by vLLM for torch 2.4.0)
RUN pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from PyPI wheels (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer==0.2.0.post2 -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install sgl-kernel from PyPI (version matching SGLang 0.3.4.post2)
RUN pip3 install sgl-kernel==0.3.4.post2

# Install vLLM 0.6.3.post1 without dependencies to avoid conflicts
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies (from requirements-cuda.txt and requirements-common.txt)
RUN pip3 install \
    "psutil" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "requests>=2.26.0" \
    "tqdm" \
    "py-cpuinfo" \
    "transformers>=4.45.2" \
    "tokenizers>=0.19.1" \
    "protobuf" \
    "fastapi>=0.107.0,!=0.113.*,!=0.114.0" \
    "aiohttp" \
    "openai>=1.40.0" \
    "uvicorn[standard]" \
    "pydantic>=2.9" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "partial-json-parser" \
    "pyzmq" \
    "msgspec" \
    "gguf==0.10.0" \
    "importlib_metadata" \
    "mistral_common[opencv]>=1.4.4" \
    "pyyaml" \
    "einops" \
    "compressed-tensors==0.6.0" \
    "ray>=2.9" \
    "nvidia-ml-py"

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

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang from source in editable mode
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Set TORCH_CUDA_ARCH_LIST for H100 GPUs
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Clean pip cache
RUN pip3 cache purge

# Verify installation
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists with content:" && \
    cat /opt/sglang_commit.txt

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]