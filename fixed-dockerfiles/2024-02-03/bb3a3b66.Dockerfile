# Dockerfile for SGLang commit bb3a3b6675b1844a13ebe368ad693f3dc75b315b
# Date: February 3, 2024
# Architecture: linux/amd64 (GPU)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hardcoded commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Verify Python version (base image should have 3.10)
RUN python --version

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch explicitly (though base image should have it)
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.2.5 without dependencies to avoid torch conflicts
RUN pip install vllm==0.2.5 --no-deps

# Install vLLM dependencies manually (based on vLLM 0.2.5 requirements.txt)
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

# Install xformers compatible with torch 2.1.x
RUN pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Build flashinfer v0.1.2 from source (no wheel available for this version)
RUN pip install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.1.2 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# Clone SGLang at the specific commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# Verify commit SHA (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="bb3a3b6675b1844a13ebe368ad693f3dc75b315b" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: Expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install other dependencies that SGLang needs (from pyproject.toml srt section)
RUN pip install \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    uvloop \
    uvicorn \
    zmq \
    interegular \
    lark \
    numba \
    pydantic \
    referencing \
    diskcache \
    cloudpickle \
    pillow \
    requests \
    "openai>=1.0" \
    anthropic

# Install SGLang package from python/ subdirectory
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Final verification of all imports
RUN python -c "import torch; print(f'torch: {torch.__version__}')" && \
    python -c "import sglang; print('SGLang import OK')" && \
    python -c "import flashinfer; print('flashinfer OK')" && \
    python -c "import vllm; print('vLLM OK')" && \
    python -c "import xformers; print(f'xformers: {xformers.__version__}')"

WORKDIR /sgl-workspace
CMD ["python", "-c", "import sglang; print('SGLang loaded successfully')"]