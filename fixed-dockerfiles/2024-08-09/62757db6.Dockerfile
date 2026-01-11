# Fixed Dockerfile for SGLang commit 62757db6f0f09a6dff15b1ee1ac3029602951509 (2024-08-09)
# vLLM 0.5.4 requires torch 2.4.0
ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        git curl wget sudo \
        build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
        ccache cmake ninja-build \
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
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 (required by vLLM 0.5.4)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4
RUN pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.4 with --no-deps to avoid torch version conflicts
RUN pip3 install vllm==0.5.4 --no-deps

# Install vLLM dependencies (manually resolved to avoid conflicts)
RUN pip3 install \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    fastapi \
    aiohttp \
    openai \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.3" \
    "outlines>=0.0.43,<0.1" \
    typing_extensions \
    "filelock>=3.10.4" \
    pyzmq \
    ray \
    nvidia-ml-py

# Install flashinfer from pre-built wheels for torch 2.4 + CUDA 12.1
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# HARDCODE the commit SHA - this Dockerfile is specific to commit 62757db6f0f09a6dff15b1ee1ac3029602951509
ENV SGLANG_COMMIT=62757db6f0f09a6dff15b1ee1ac3029602951509

# Clone SGLang repo and checkout EXACT commit (hardcoded, not variable)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 62757db6f0f09a6dff15b1ee1ac3029602951509

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="62757db6f0f09a6dff15b1ee1ac3029602951509" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from checked-out source
# Note: pyproject.toml is in python/ subdirectory at this commit
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install triton-nightly to avoid version conflicts
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Set TORCH_CUDA_ARCH_LIST for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Reset to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]