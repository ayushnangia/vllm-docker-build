# Fixed Dockerfile for SGLang commit 6f560c761b2fc2f577682d0cfda62630f37a3bb0
# Date: 2024-01-30 (PR #117 - "Improve the control of streaming and improve the first token latency in streaming")
#
# Requirements from pyproject.toml:
# - vllm >= 0.2.5 (which requires torch >= 2.1.1)
# - Python 3.8+
# - No flashinfer mentioned (very early commit)
# - No sgl-kernel mentioned

# Use PyTorch base image for early 2024 commits (as per guidelines)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.1.2 with CUDA 12.1 (already in base image, but ensure correct version)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.2.5 with --no-deps to avoid pulling wrong torch version
RUN pip install vllm==0.2.5 --no-deps

# Install vLLM dependencies manually (from vLLM 0.2.5 requirements.txt)
# Note: torch already installed, xformers needs exact version for torch 2.1.x
RUN pip install \
    "transformers>=4.36.0" \
    "xformers==0.0.23.post1" \
    "numpy" \
    "sentencepiece" \
    "ray>=2.5.1" \
    "pandas" \
    "pyarrow" \
    "fastapi" \
    "uvicorn[standard]" \
    "pydantic==1.10.13" \
    "aioprometheus[starlette]" \
    "ninja" \
    "psutil" \
    "nvidia-ml-py"

# Install additional dependencies from SGLang pyproject.toml (srt optional deps)
RUN pip install \
    "aiohttp" \
    "rpyc" \
    "uvloop" \
    "pyzmq" \
    "interegular" \
    "lark" \
    "numba" \
    "diskcache" \
    "cloudpickle" \
    "pillow" \
    "openai>=1.0" \
    "anthropic"

# HARDCODE the commit SHA (1st occurrence - ENV)
ENV SGLANG_COMMIT=6f560c761b2fc2f577682d0cfda62630f37a3bb0

# Clone SGLang and checkout EXACT commit (2nd occurrence - git checkout)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 6f560c761b2fc2f577682d0cfda62630f37a3bb0

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence - verification)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6f560c761b2fc2f577682d0cfda62630f37a3bb0" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from checked-out source
# Note: Using "python[all]" syntax as shown in original requirements
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')"

# Verify vLLM can be imported
RUN python3 -c "import vllm; print('vLLM import OK')"

# Verify torch version is correct
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}'); assert torch.__version__.startswith('2.1'), 'Wrong torch version'"

# Set working directory for runtime
WORKDIR /sgl-workspace/sglang

# Default entrypoint (can be overridden)
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]