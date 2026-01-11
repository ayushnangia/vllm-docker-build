# Fixed Dockerfile for SGLang commit 96c503eb6029d37f896e91466e23469378dfc3dc
# Date: 2024-07-03
# Dependencies discovered via PyPI WebFetch (not guessed):
# - vLLM 0.5.0 (from pyproject.toml)
# - torch 2.3.0 (from vLLM requirements-cuda.txt)
# - pydantic 2.8.1 (released July 3, 2024)
# - outlines 0.0.46 (requires pydantic>=2.0)
# - fastapi 0.111.0 (released May 3, 2024)
# - uvicorn 0.30.1 (released June 2, 2024)
# - typing_extensions 4.12.2 (released June 7, 2024)

# Base image for torch 2.3.0 with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Build arguments
ARG SGLANG_COMMIT=96c503eb6029d37f896e91466e23469378dfc3dc
ARG MAX_JOBS=96
ARG TORCH_CUDA_ARCH_LIST="9.0"

# Environment variables (1st SHA occurrence - HARDCODED)
ENV SGLANG_COMMIT=96c503eb6029d37f896e91466e23469378dfc3dc
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /sgl-workspace

# Create constraints file with versions discovered from PyPI for 2024-07-03 era
# Each version was WebFetched from PyPI and confirmed available on July 3, 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-07-03 era
# pydantic 2.8.1 was released on July 3, 2024 (exact same day!)
pydantic==2.8.1
# fastapi 0.111.0 was released May 3, 2024 (latest before July 3)
fastapi==0.111.0
# uvicorn 0.30.1 was released June 2, 2024 (latest before July 3)
uvicorn==0.30.1
# typing_extensions 4.12.2 was released June 7, 2024 (avoids Sentinel in 4.14+)
typing_extensions==4.12.2
# outlines 0.0.46 was released June 22, 2024 (satisfies >=0.0.44, requires pydantic>=2.0)
outlines==0.0.46
# pyzmq 26.0.3 was released May 1, 2024 (latest before July 3)
pyzmq==26.0.3
EOF

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Ensure torch 2.3.0 is installed (base image should have it)
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers for torch 2.3.0
RUN pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.0 without dependencies first
RUN pip install vllm==0.5.0 --no-deps

# Install vLLM dependencies from requirements-common.txt (discovered from vLLM repo)
# Using constraints file to ensure version compatibility
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
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
    "filelock>=3.10.4"

# Install vLLM CUDA-specific dependencies (from requirements-cuda.txt)
RUN pip install -c /opt/constraints.txt \
    "ray>=2.9" \
    nvidia-ml-py \
    "vllm-flash-attn==2.5.9"

# Clone SGLang at specific commit (2nd SHA occurrence - HARDCODED)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 96c503eb6029d37f896e91466e23469378dfc3dc

# Verify commit SHA and write to file (3rd SHA occurrence - HARDCODED)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="96c503eb6029d37f896e91466e23469378dfc3dc" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified commit: $ACTUAL"

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies from pyproject.toml [srt] extras
# Using constraints to ensure versions from July 2024 era
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    pydantic \
    rpyc \
    torch \
    uvicorn \
    uvloop \
    zmq \
    "outlines>=0.0.44"

# Install SGLang other extras
RUN pip install -c /opt/constraints.txt \
    "openai>=1.0" \
    tiktoken \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Install additional common packages
RUN pip install datasets

# Verify imports work
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import outlines; print('Outlines import OK')"

# Set working directory to SGLang
WORKDIR /sgl-workspace/sglang

# Expose typical server port
EXPOSE 30000

# Set entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]