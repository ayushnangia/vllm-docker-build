# SGLang Docker image for commit 73fa2d49d539fd67548b0458a365528d3e3b6edc
# Date: 2024-08-09
# Requirements: vLLM 0.5.4, torch 2.4.0

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies and timezone
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        git \
        curl \
        wget \
        sudo \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        ccache \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=73fa2d49d539fd67548b0458a365528d3e3b6edc

# Pre-install torch 2.4.0 with CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.4 with --no-deps to avoid dependency conflicts
RUN pip3 install --no-cache-dir vllm==0.5.4 --no-deps

# Install vLLM dependencies (based on vLLM 0.5.4 requirements-cuda.txt and requirements-common.txt)
RUN pip3 install --no-cache-dir \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "sentencepiece" \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1" \
    "cmake>=3.21" \
    "ninja" \
    "psutil" \
    "numpy<2.0.0" \
    "requests" \
    "tqdm" \
    "py-cpuinfo" \
    "fastapi" \
    "aiohttp" \
    "openai" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.3" \
    "outlines>=0.0.43,<0.1" \
    "torchvision==0.19" \
    "typing_extensions" \
    "filelock>=3.10.4" \
    "pyzmq"

# Install flashinfer from wheel repository (available for torch 2.4 + CUDA 12.1)
RUN pip3 install --no-cache-dir flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 73fa2d49d539fd67548b0458a365528d3e3b6edc

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="73fa2d49d539fd67548b0458a365528d3e3b6edc" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install other SGLang dependencies that aren't part of vLLM
RUN pip3 install --no-cache-dir \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "packaging" \
    "python-multipart" \
    "uvloop" \
    "zmq"

# Install SGLang from python subdirectory with all extras
WORKDIR /sgl-workspace/sglang
RUN pip3 install --no-cache-dir -e "python[all]"

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Clean up
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]