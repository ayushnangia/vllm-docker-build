# Fixed Dockerfile for SGLang commit ab4a83b25909aa98330b838a224e4fe5c943e483
# Date: 2024-09-05
# SGLang version: 0.3.0
# vLLM version: 0.5.5
# torch version: 2.4.0 (required by vLLM 0.5.5)
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo \
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
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip \
    && cd .. && rm -rf Python-3.10.14*

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 support (required by vLLM 0.5.5)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from PyPI wheel (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vllm==0.5.5 with --no-deps to avoid torch version conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies (from vLLM 0.5.5 requirements-cuda.txt, excluding torch)
RUN pip3 install \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "torchvision==0.19" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1" \
    "psutil" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "requests" \
    "tqdm" \
    "py-cpuinfo" \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    "protobuf" \
    "fastapi" \
    "aiohttp" \
    "openai>=1.0" \
    "uvicorn[standard]" \
    "pydantic>=2.8" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "pyzmq" \
    "msgspec" \
    "librosa" \
    "soundfile" \
    "gguf==0.9.1" \
    "importlib_metadata"

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=ab4a83b25909aa98330b838a224e4fe5c943e483

# Clone SGLang repo and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ab4a83b25909aa98330b838a224e4fe5c943e483

# VERIFY commit matches expected value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ab4a83b25909aa98330b838a224e4fe5c943e483" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source with all dependencies (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install additional dependencies that might be needed
RUN pip3 install numpy packaging

# Verify all critical installations
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Clean up pip cache
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]