# Dockerfile for SGLang commit 45d6592d4053fe8b2b8dc9440f64c900de040d09
# Date: 2024-02-03
# SGLang version: 0.1.11 (requires vllm>=0.2.5)
# Architecture: linux/amd64 (GPU)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HARDCODED commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=45d6592d4053fe8b2b8dc9440f64c900de040d09

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Pre-install torch (already in base image, but ensure correct version)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.1.2 (vLLM 0.2.5 requires xformers >= 0.0.23)
RUN pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.2.5 WITHOUT dependencies to avoid torch version conflicts
RUN pip install vllm==0.2.5 --no-deps

# Install vLLM dependencies from vLLM 0.2.5 requirements.txt (discovered from exploration)
RUN pip install \
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

# Clone SGLang repository (HARDCODED SHA occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 45d6592d4053fe8b2b8dc9440f64c900de040d09

# Verify correct commit (HARDCODED SHA occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="45d6592d4053fe8b2b8dc9440f64c900de040d09" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove vllm since we already installed it
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install additional SGLang dependencies from pyproject.toml
RUN pip install \
    aiohttp \
    rpyc \
    uvloop \
    pyzmq \
    interegular \
    lark \
    numba \
    referencing \
    jsonschema \
    diskcache \
    cloudpickle \
    pillow \
    "openai>=1.0" \
    anthropic \
    requests

# Install SGLang package (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')"

WORKDIR /sgl-workspace
CMD ["python3", "-c", "import sglang; print('SGLang loaded successfully')"]