# Fixed Dockerfile for SGLang commit 2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0
# Commit date: 2025-01-18 (TIME-FREEZE to June 2024 era for dependencies)
# Requirements from pyproject.toml:
#   - vllm>=0.6.3.post1,<=0.6.4.post1 (requires torch 2.4.0)
#   - flashinfer==0.1.6
#   - sgl-kernel>=0.0.2.post14 (using 0.1.0 from PyPI)
#   - outlines>=0.0.44,<0.1.0 (requires pydantic v2)

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

# Create constraints file with versions discovered from PyPI for June 2024 era
# These versions are TIME-FROZEN to June 2024 when outlines 0.0.44 was released
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for June 2024 era
# outlines 0.0.44 was released June 14, 2024
# These versions are compatible with pydantic v2 (required by outlines>=0.0.44)
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
pydantic-core==2.18.4
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.0.3
EOF

# Install vLLM 0.6.3.post1 with --no-deps to avoid dependency conflicts
RUN pip install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with constraints
# Based on vllm/requirements-common.txt and requirements-cuda.txt
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    'numpy<2.0.0' \
    'requests>=2.26.0' \
    tqdm \
    py-cpuinfo \
    'transformers>=4.45.2' \
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
    'xformers==0.0.27.post2'

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

# Install flashinfer from wheels (0.1.6 is available)
RUN pip install flashinfer==0.1.6 --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# Install sgl-kernel from PyPI (0.1.0 is available, meets >=0.0.2.post14 requirement)
RUN pip install sgl-kernel==0.1.0

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
    'xgrammar>=0.1.6'

# Install SGLang core dependencies
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Install optional dependencies for full functionality
RUN pip install \
    'openai>=1.0' \
    tiktoken \
    'anthropic>=0.20.0' \
    'litellm>=1.0.0'

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
    python3 -c "import sgl_kernel; print('sgl-kernel imported successfully')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} OK')" && \
    python3 -c "import outlines; print('outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'pydantic {pydantic.VERSION} imported successfully')"

# Verify commit file exists and is correct
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit verification file contents:" && \
    cat /opt/sglang_commit.txt

# Default entrypoint (can be overridden)
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]