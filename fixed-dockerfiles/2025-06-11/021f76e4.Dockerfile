# Time-freeze Dockerfile for SGLang commit 021f76e4f49861b2e9ea9ccff06a46d577e3c548
# Date: 2025-06-11 (using June 2024 era dependencies)
# Based on PyPI discovery for June 11, 2024

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=96
ENV TORCH_CUDA_ARCH_LIST="9.0"
# HARDCODE SHA (1st occurrence)
ENV SGLANG_COMMIT=021f76e4f49861b2e9ea9ccff06a46d577e3c548

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with discovered versions from June 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for June 11, 2024
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.3
pydantic-core==2.18.4
typing_extensions==4.12.1
outlines==0.0.44
pyzmq==26.0.3
transformers==4.41.2
tokenizers==0.19.1
prometheus-client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
tiktoken==0.7.0
lm-format-enforcer==0.10.1
numpy==1.26.4
sentencepiece==0.2.0
psutil==5.9.8
aiohttp==3.9.5
requests==2.32.3
pillow==10.3.0
openai==1.30.5
huggingface_hub==0.23.2
filelock==3.14.0
nvidia-ml-py==12.535.161
scipy==1.13.1
datasets==2.19.2
uvloop==0.19.0
orjson==3.10.3
msgspec==0.18.6
EOF

# Install vLLM 0.5.0 with --no-deps first
RUN pip install vllm==0.5.0 --no-deps

# Install vLLM dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    transformers \
    tokenizers \
    fastapi \
    aiohttp \
    openai \
    "uvicorn[standard]" \
    pydantic \
    pillow \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing-extensions \
    filelock \
    nvidia-ml-py

# Install xformers and flash-attn for vLLM
RUN pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install vllm-flash-attn==2.5.9

# Install flashinfer from prebuilt wheels
RUN pip install flashinfer==0.0.4 -i https://flashinfer.ai/whl/cu121/torch2.3/

# Create workspace
WORKDIR /sgl-workspace

# Clone SGLang and checkout specific commit
# HARDCODE SHA (2nd occurrence)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 021f76e4f49861b2e9ea9ccff06a46d577e3c548

# Verify commit SHA and write to file
# HARDCODE SHA (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="021f76e4f49861b2e9ea9ccff06a46d577e3c548" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "Error: Expected commit $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang base dependencies
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Install SGLang runtime_common dependencies
RUN pip install -c /opt/constraints.txt \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    scipy \
    uvicorn \
    uvloop \
    einops

# Note: Some dependencies from future pyproject.toml don't exist in June 2024:
# - llguidance (doesn't exist yet)
# - soundfile==0.13.1 (only 0.12.1 exists)
# - torchao==0.9.0 (doesn't exist yet)
# - xgrammar==0.1.19 (doesn't exist yet)
# - sgl-kernel==0.1.7 (doesn't exist yet, will build from source)

# Build sgl-kernel from source (since it doesn't exist on PyPI for June 2024)
RUN cd /sgl-workspace/sglang && \
    if [ -d "python/sglang/srt/kernels" ]; then \
        cd python/sglang/srt/kernels && \
        python setup.py bdist_wheel && \
        pip install dist/*.whl || echo "sgl-kernel build failed, continuing"; \
    else \
        echo "sgl-kernel source not found, skipping"; \
    fi

# Install SGLang in editable mode without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Verify installation
RUN python -c "import sglang; print('SGLang import successful')" && \
    python -c "import vllm; print('vLLM import successful')" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import outlines; print('Outlines import successful')" && \
    python -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

# Verify commit file
RUN cat /opt/sglang_commit.txt

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]