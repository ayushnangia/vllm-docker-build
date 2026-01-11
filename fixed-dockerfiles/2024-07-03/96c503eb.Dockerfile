# Fixed Dockerfile for SGLang commit 96c503eb6029d37f896e91466e23469378dfc3dc
# Date: 2024-07-03
# Dependencies: vLLM 0.5.0, torch 2.3.0, xformers 0.0.26.post1

# Use pytorch base image for torch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# HARDCODED commit SHA (1 of 3 occurrences)
ENV SGLANG_COMMIT=96c503eb6029d37f896e91466e23469378dfc3dc

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Pre-install torch 2.3.0 (already in base image, but ensure it's the right version)
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers for torch 2.3.0
RUN pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.0 with --no-deps to avoid pulling wrong torch version
RUN pip install vllm==0.5.0 --no-deps

# Install vLLM dependencies manually (based on requirements-cuda.txt from vLLM 0.5.0)
RUN pip install \
    nvidia-ml-py \
    ray>=2.9 \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    "transformers>=4.40.0" \
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
    "lm-format-enforcer==0.10.1" \
    "outlines>=0.0.43" \
    typing_extensions \
    "filelock>=3.10.4" \
    psutil \
    packaging \
    "vllm-flash-attn==2.5.9"

# Clone SGLang repo and checkout EXACT commit (2 of 3 occurrences)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 96c503eb6029d37f896e91466e23469378dfc3dc

# VERIFY the checkout - compare against HARDCODED expected value (3 of 3 occurrences)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="96c503eb6029d37f896e91466e23469378dfc3dc" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from checked-out source
# Note: pyproject.toml is in python/ subdirectory for this commit
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Install datasets as in original Dockerfile
RUN pip install datasets

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
