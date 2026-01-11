# Fixed Dockerfile for SGLang
# Commit: da19434c (April 25, 2024)
# Date: 2024-04-25
# PyTorch 2.2.1 (required by vLLM 0.4.1), CUDA 12.1

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV SGLANG_COMMIT=da19434c2f3cbe4f367f84993da0bcbd84efb6ba

# Update and install dependencies
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

# Install Python packages with proper constraints
# Based on discovered versions from PyPI for April 25, 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies with versions discovered from PyPI for 2024-04-25
fastapi==0.110.2
uvicorn==0.29.0
pydantic==2.7.1
typing_extensions==4.11.0
outlines==0.0.34
pyzmq==26.0.2
tiktoken==0.6.0
lm-format-enforcer==0.9.8
prometheus_client==0.20.0
EOF

# Install vLLM 0.4.1 dependencies first
RUN pip install --upgrade pip setuptools wheel

# Install vLLM 0.4.1 without dependencies
RUN pip install vllm==0.4.1 --no-deps

# Install vLLM dependencies from requirements-common.txt
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
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "prometheus_client>=0.18.0" \
    tiktoken==0.6.0 \
    lm-format-enforcer==0.9.8 \
    outlines==0.0.34 \
    typing_extensions \
    "filelock>=3.10.4"

# Install vLLM CUDA-specific dependencies
RUN pip install -c /opt/constraints.txt \
    "ray>=2.9" \
    nvidia-ml-py \
    "vllm-nccl-cu12>=2.18,<2.19" \
    xformers==0.0.25

# Clone SGLang at the specific commit (1st occurrence of SHA)
RUN git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git checkout da19434c2f3cbe4f367f84993da0bcbd84efb6ba

# Verify correct commit SHA (2nd occurrence of SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    if [ "$ACTUAL" != "$SGLANG_COMMIT" ]; then \
        echo "ERROR: Git checkout failed. Expected $SGLANG_COMMIT, got $ACTUAL"; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies from pyproject.toml [srt] extras
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    torch \
    uvloop \
    uvicorn \
    pyzmq \
    "vllm>=0.3.3" \
    interegular \
    pydantic \
    pillow \
    "outlines>=0.0.27"

# Install additional SGLang core dependencies
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm

# Sanity check - verify all key imports work
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import outlines; print('Outlines import OK')" && \
    python3 -c "import pydantic; print('Pydantic import OK')" && \
    python3 -c "import fastapi; print('FastAPI import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Final verification of commit SHA (3rd occurrence of SHA)
RUN test "$(cat /opt/sglang_commit.txt)" = "da19434c2f3cbe4f367f84993da0bcbd84efb6ba" || \
    (echo "ERROR: Commit SHA mismatch in final verification" && exit 1)

# Set the working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]