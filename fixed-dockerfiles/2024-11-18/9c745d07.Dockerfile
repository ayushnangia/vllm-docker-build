# Fixed Dockerfile for SGLang commit 9c745d078e29e153a64300bd07636c7c9c1c42d5
# Date: 2024-11-18
# Requirements: vLLM 0.6.3.post1, torch 2.4.0, flashinfer >= 0.1.6

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
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

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 for CUDA 12.1 (required by vLLM 0.6.3.post1)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install torchvision aligned with torch (required by vLLM)
RUN pip3 install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4.0
RUN pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.6.3.post1 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies (from requirements-common.txt and requirements-cuda.txt)
RUN pip3 install \
    "transformers>=4.45.2" \
    "tokenizers>=0.19.1" \
    "numpy<2.0.0" \
    "pydantic>=2.9" \
    "fastapi>=0.107.0,!=0.113.*,!=0.114.0" \
    "openai>=1.40.0" \
    "uvicorn[standard]" \
    "lm-format-enforcer==0.10.6" \
    "gguf==0.10.0" \
    "compressed-tensors==0.6.0" \
    "outlines>=0.0.43,<0.1" \
    "tiktoken>=0.6.0" \
    "mistral_common[opencv]>=1.4.4" \
    "einops" \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "prometheus-client>=0.20.0" \
    "psutil" \
    "aiohttp" \
    "pillow" \
    "python-multipart" \
    "requests" \
    "tqdm" \
    "orjson" \
    "packaging" \
    "huggingface_hub" \
    "pyzmq>=25.1.2" \
    "modelscope" \
    "decord" \
    "hf_transfer" \
    "interegular"

# Install flashinfer from wheels (available for torch 2.4 + CUDA 12.1)
# SGLang requires flashinfer >= 0.1.6, using 0.2.0.post2
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install torchao (from SGLang pyproject.toml)
RUN pip3 install torchao

# Install other useful packages for MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=9c745d078e29e153a64300bd07636c7c9c1c42d5

# Clone SGLang repo and checkout EXACT commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 9c745d078e29e153a64300bd07636c7c9c1c42d5

# VERIFY the checkout - compare against HARDCODED expected value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9c745d078e29e153a64300bd07636c7c9c1c42d5" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from checked-out source
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace Triton with nightly version (common pattern in SGLang dockerfiles)
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Clear pip cache
RUN pip3 cache purge

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.version.cuda} OK')"

# Set interactive mode back
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]