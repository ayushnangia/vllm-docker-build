# Fixed Dockerfile for SGLang commit 33b242df303e03886835d08a583fefe979a3ee88
# Date: 2024-05-11
# Requires: torch 2.3.0, vllm>=0.4.2, flashinfer>=0.0.4
# CRITICAL FIX: pyproject.toml is in python/ subdirectory for this commit

# Use pytorch base image for torch 2.3.0 (more stable than tritonserver for this era)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# HARDCODE the commit SHA (do not use ARG to avoid forgotten --build-arg issues)
ENV SGLANG_COMMIT=33b242df303e03886835d08a583fefe979a3ee88

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Pre-install torch 2.3.0 with CUDA 12.1 (ensure correct version)
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.4.2 with --no-deps to avoid torch version conflicts
RUN pip install vllm==0.4.2 --no-deps

# Install vLLM dependencies manually (from vLLM requirements-cuda.txt)
RUN pip install \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "sentencepiece" \
    "psutil" \
    "numpy" \
    "requests" \
    "py-cpuinfo" \
    "transformers>=4.40.0" \
    "tokenizers>=0.19.1" \
    "fastapi" \
    "openai" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken==0.6.0" \
    "lm-format-enforcer==0.9.8" \
    "outlines==0.0.34" \
    "typing_extensions" \
    "filelock>=3.10.4"

# Install xformers 0.0.26.post1 for torch 2.3.0
RUN pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from wheel (available for torch 2.3 + CUDA 12.1)
# Using version 0.0.8 which is stable and meets the >=0.0.4 requirement
RUN pip install flashinfer==0.0.8 -i https://flashinfer.ai/whl/cu121/torch2.3/

# Clone SGLang repository and checkout EXACT commit (hardcoded SHA - occurrence 1 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 33b242df303e03886835d08a583fefe979a3ee88

# VERIFY the checkout - compare against HARDCODED expected value (occurrence 2 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="33b242df303e03886835d08a583fefe979a3ee88" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# CRITICAL: Patch pyproject.toml in the CORRECT location (python/ subdirectory)
# Remove already-installed dependencies to avoid version conflicts
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"torch",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"xformers[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/,,/,/g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/,\]/]/g' /sgl-workspace/sglang/python/pyproject.toml

# Install additional SGLang dependencies
RUN pip install \
    "aiohttp" \
    "rpyc" \
    "uvloop" \
    "zmq" \
    "interegular" \
    "pillow" \
    "packaging" \
    "tqdm"

# CRITICAL: Install SGLang from checked-out source
# Note: pyproject.toml is in python/ subdirectory for this commit
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Install datasets as mentioned in original Dockerfile
RUN pip install datasets

# Verify commit is correct (occurrence 3 of 3 - verification)
RUN cd /sgl-workspace/sglang && \
    COMMIT_CHECK=$(git rev-parse HEAD) && \
    test "$COMMIT_CHECK" = "33b242df303e03886835d08a583fefe979a3ee88" || (echo "ERROR: Wrong commit!" && exit 1)

# Verify all imports work correctly
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm import OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Set entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]