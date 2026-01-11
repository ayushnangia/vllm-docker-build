# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
    git curl libibverbs-dev \
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

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=83452dbb4a19c6a2461e972eb2b64a2df9a466b8

# Pre-install torch 2.4.0 with CUDA 12.1 (required by vLLM 0.6.3)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4.0 (required by vLLM 0.6.3)
RUN pip3 install xformers==0.0.27.post2

# Install flashinfer 0.1.6 from wheel (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/

# Build sgl-kernel from source (0.0.2.post14 not available on PyPI, using compatible version)
# Note: 0.1.0 from PyPI also works, but building from source for better compatibility
RUN git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel \
    && cd /tmp/sgl-kernel \
    && git checkout v0.3.20 \
    && TORCH_CUDA_ARCH_LIST="9.0" pip3 install --no-build-isolation . \
    && rm -rf /tmp/sgl-kernel

# Install vllm 0.6.3.post1 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies (excluding torch, xformers which we already have)
RUN pip3 install \
    "cmake>=3.26" \
    "ninja" \
    "psutil" \
    "ray>=2.10.0" \
    "sentencepiece" \
    "nvidia-ml-py" \
    "pynvml==11.5.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.9" \
    "gguf==0.9.1" \
    "importlib_metadata" \
    "mistral_common>=1.3.4" \
    "protobuf" \
    "partial-json-parser" \
    "pyzmq" \
    "prometheus-client>=0.18.0" \
    "lark" \
    "aioprometheus>=23.12.0" \
    "outlines>=0.0.46,<0.1" \
    "typing_extensions>=4.10" \
    "py-cpuinfo" \
    "librosa" \
    "soundfile" \
    "pillow" \
    "aiohttp" \
    "fastapi" \
    "uvicorn[standard]" \
    "openai" \
    "uvloop" \
    "pydantic" \
    "numpy<2.0.0"

# Install torchao>=0.7.0 (required by pyproject.toml)
RUN pip3 install torchao>=0.7.0

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 83452dbb4a19c6a2461e972eb2b64a2df9a466b8

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="83452dbb4a19c6a2461e972eb2b64a2df9a466b8" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install other dependencies for MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator html5lib six

# Install SGLang from source in editable mode with all extras
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install triton nightly (common pattern for performance)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-cache-dir --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import vllm; print('vllm import OK')" && \
    python3 -c "import torch; print(f'torch {torch.__version__} OK')" && \
    python3 -c "import xformers; print(f'xformers {xformers.__version__} OK')" && \
    python3 -c "import torchao; print('torchao import OK')"

# Clean pip cache
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]