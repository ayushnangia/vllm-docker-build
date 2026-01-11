# SGLang Dockerfile for commit 5d264a90ac5154d8e368ee558337dd3dd92e720b
# Date: 2024-07-14
# SGLang version: 0.1.20
# vLLM version: 0.5.1 (requires torch 2.3.0)

# Base image for torch 2.3.x with CUDA 12.1 (using pytorch base which has Python already)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Pre-install torch 2.3.0 (vLLM 0.5.1 requirement) with CUDA 12.1
# Torch is already in the base image, but we ensure correct version
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.3.0
RUN pip install xformers==0.0.26.post1

# Install vLLM 0.5.1 with --no-deps to avoid pulling wrong torch version
RUN pip install vllm==0.5.1 --no-deps

# Install vLLM dependencies manually (from vLLM 0.5.1 requirements-cuda.txt)
RUN pip install \
    "ray>=2.9" \
    "nvidia-ml-py" \
    "vllm-flash-attn==2.5.9" \
    "cmake>=3.21" \
    "ninja" \
    "psutil" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "requests" \
    "tqdm" \
    "py-cpuinfo" \
    "transformers>=4.42.0" \
    "tokenizers>=0.19.1" \
    "fastapi" \
    "aiohttp" \
    "openai" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.1" \
    "outlines>=0.0.43" \
    "typing_extensions" \
    "filelock>=3.10.4"

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=5d264a90ac5154d8e368ee558337dd3dd92e720b

# Clone SGLang and checkout EXACT commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 5d264a90ac5154d8e368ee558337dd3dd92e720b

# VERIFY commit - compare against HARDCODED value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="5d264a90ac5154d8e368ee558337dd3dd92e720b" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source in editable mode
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install triton nightly to replace the default triton
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set working directory and entrypoint
WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]