# Fixed Dockerfile for SGLang commit cd6872334e9ead684049b8fccd5f2dac9433b1b4
# Date: 2024-01-30 (Very Early era - January 2024)
# SGLang version: 0.1.9
# Requirements: vllm>=0.2.5, torch>=2.1.1 (via vLLM)
# No flashinfer or sgl-kernel required at this early stage

# Use pytorch base image for torch 2.1.x with CUDA 12.1 (early 2024 commit)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# CRITICAL: Pre-install torch 2.1.2 with CUDA 12.1 to ensure correct version
# (vLLM might pull different version if we don't pin it first)
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.1.x (per guidelines compatibility matrix)
RUN pip3 install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.2.5 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.2.5 --no-deps

# Install vLLM dependencies manually (excluding torch and xformers which we already installed)
# Based on vLLM 0.2.5 requirements.txt we checked earlier
RUN pip3 install \
    ninja \
    psutil \
    "ray>=2.5.1" \
    pandas \
    pyarrow \
    sentencepiece \
    numpy \
    "transformers>=4.36.0" \
    fastapi \
    "uvicorn[standard]" \
    "pydantic==1.10.13" \
    "aioprometheus[starlette]"

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=cd6872334e9ead684049b8fccd5f2dac9433b1b4

# Clone SGLang repo and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout cd6872334e9ead684049b8fccd5f2dac9433b1b4

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="cd6872334e9ead684049b8fccd5f2dac9433b1b4" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install additional SGLang dependencies from pyproject.toml[srt]
RUN pip3 install \
    aiohttp \
    rpyc \
    uvloop \
    zmq \
    interegular \
    lark \
    numba \
    diskcache \
    cloudpickle \
    pillow \
    requests \
    "openai>=1.0" \
    anthropic

# Install SGLang from checked-out source
# CRITICAL: pyproject.toml is in python/ subdirectory for this early commit
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Verify installations
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"

# Final verification - SGLang import
RUN python3 -c "import sglang; print('SGLang import OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Entry point (sglang.launch_server module exists at this commit)
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]