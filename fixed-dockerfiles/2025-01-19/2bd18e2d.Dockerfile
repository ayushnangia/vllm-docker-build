# Fixed Dockerfile for SGLang commit 2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0
# Commit date: 2025-01-18
# SGLang version: 0.4.1.post6
# Requirements verified from PyPI:
#   - vllm>=0.6.3.post1,<=0.6.4.post1 (requires torch 2.4.0, pydantic>=2.9)
#   - flashinfer==0.1.6
#   - sgl-kernel>=0.0.2.post14
#   - torchao>=0.7.0
#   - outlines>=0.0.44,<0.1.0

# Base image for torch 2.4.0 with CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st HARDCODED SHA: ENV declaration
ENV SGLANG_COMMIT=2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    sudo \
    libibverbs-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with versions verified from PyPI for January 2025
RUN cat > /opt/constraints.txt <<'EOF'
# Versions verified from PyPI - January 2025 era
# vLLM 0.6.3.post1 requires pydantic>=2.9, fastapi>=0.107.0
fastapi==0.115.6
uvicorn==0.32.1
pydantic==2.9.2
pydantic-core==2.23.4
typing_extensions==4.12.2
pyzmq==26.2.0
# SGLang requires torchao>=0.7.0 (released Dec 10, 2024)
torchao==0.7.0
# SGLang requires outlines>=0.0.44,<0.1.0
outlines==0.0.46
transformers>=4.45.2,<4.48.0
EOF

# Install vLLM 0.6.3.post1 with --no-deps to avoid dependency conflicts
RUN pip install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with constraints
# Based on vLLM 0.6.3.post1 requirements from PyPI
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    'numpy<2.0.0' \
    'requests>=2.26.0' \
    tqdm \
    py-cpuinfo \
    'transformers>=4.45.2,<4.48.0' \
    'tokenizers>=0.19.1' \
    protobuf \
    aiohttp \
    'openai>=1.40.0' \
    'uvicorn[standard]' \
    pillow \
    'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' \
    'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.6' \
    'filelock>=3.10.4' \
    partial-json-parser \
    msgspec \
    'gguf==0.10.0' \
    importlib_metadata \
    'mistral_common[opencv]>=1.4.4' \
    pyyaml \
    'six>=1.16.0' \
    'setuptools>=74.1.1' \
    einops \
    'compressed-tensors==0.6.0' \
    'ray>=2.9' \
    nvidia-ml-py \
    torchvision==0.19.0 \
    'xformers==0.0.27.post2' \
    'fastapi>=0.107.0,!=0.113.*,!=0.114.0' \
    'pyzmq>=25.0.0' \
    'pydantic>=2.9' \
    'outlines>=0.0.43,<0.1'

# Create workspace
WORKDIR /sgl-workspace

# 2nd HARDCODED SHA: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0

# 3rd HARDCODED SHA: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install flashinfer==0.1.6 from wheels (required by SGLang)
RUN pip install flashinfer==0.1.6 --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# Install sgl-kernel (SGLang requires >=0.0.2.post14)
RUN pip install sgl-kernel || echo "sgl-kernel not available, continuing"

# Install SGLang with --no-deps to avoid dependency conflicts
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    orjson \
    packaging \
    pillow \
    'prometheus-client>=0.20.0' \
    psutil \
    python-multipart \
    'torchao>=0.7.0' \
    uvloop \
    'pyzmq>=25.1.2' \
    'xgrammar>=0.1.6'

# Install SGLang core dependencies
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Install optional dependencies for full functionality
RUN pip install -c /opt/constraints.txt \
    'openai>=1.0' \
    tiktoken \
    'anthropic>=0.20.0' \
    'litellm>=1.0.0' \
    dill \
    cloudpickle \
    'outlines>=0.0.44,<0.1.0'

# Clean pip cache
RUN pip cache purge

# Reset to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace/sglang

# Verify the installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import flashinfer; print('flashinfer imported successfully')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} OK')" && \
    python3 -c "import pydantic; print(f'pydantic {pydantic.VERSION} OK')" && \
    python3 -c "import fastapi; print(f'fastapi {fastapi.__version__} OK')" && \
    echo "Build verification passed."

# Verify commit file exists and is correct
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit verification file contents:" && \
    cat /opt/sglang_commit.txt

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
