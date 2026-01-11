# Fixed Dockerfile for SGLang commit 2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0
# Date: 2025-01-19
# Requirements from pyproject.toml:
#   - torch 2.4.0 (required by vLLM 0.6.3.post1)
#   - flashinfer==0.1.6 (available as wheel)
#   - vllm>=0.6.3.post1,<=0.6.4.post1
#   - sgl-kernel>=0.0.2.post14 (build from source - not on PyPI)
#   - torchao>=0.7.0

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA broken on Ubuntu 20.04)
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

# Pre-install torch 2.4.0 with CUDA 12.1 (vLLM 0.6.3.post1 requires torch 2.4.0)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer 0.1.6 from wheel
RUN pip3 install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# Build sgl-kernel from source (0.0.2.post14 not on PyPI)
RUN pip3 install numpy packaging \
    && git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel \
    && cd /tmp/sgl-kernel \
    && git checkout v0.0.2.post14 \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . \
    && rm -rf /tmp/sgl-kernel

# Install vLLM with --no-deps to avoid pulling wrong torch version
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies (manually resolved for torch 2.4.0)
RUN pip3 install \
    "xformers==0.0.27.post2" \
    "nvidia-ml-py" \
    "sentencepiece" \
    "transformers>=4.36.0" \
    "ray>=2.9.0" \
    "tiktoken" \
    "msgspec" \
    "lm-format-enforcer==0.10.9" \
    "gguf==0.10.0" \
    "pillow" \
    "prometheus-client>=0.20.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "pydantic>=2.9" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.16.1" \
    "pyzmq>=25.1.2" \
    "uvloop>=0.21.0" \
    "compressed-tensors==0.8.0" \
    "aiohttp" \
    "fastapi>=0.115.4" \
    "uvicorn[standard]" \
    "openai" \
    "psutil"

# Install torchao
RUN pip3 install torchao>=0.7.0

# HARDCODE commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0" && \
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
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/, *,/,/g; s/\[,/[/g; s/,\]/]/g' python/pyproject.toml

# Install remaining runtime_common dependencies that weren't manually installed
RUN pip3 install \
    "decord" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "modelscope" \
    "orjson" \
    "python-multipart" \
    "xgrammar>=0.1.6" \
    "requests" \
    "tqdm" \
    "numpy" \
    "IPython" \
    "setproctitle" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} OK')"

# Clean up pip cache
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

# Set working directory for runtime
WORKDIR /sgl-workspace/sglang

# Entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]