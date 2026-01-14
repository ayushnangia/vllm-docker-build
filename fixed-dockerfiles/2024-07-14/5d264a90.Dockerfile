# SGLang Docker Build for July 14, 2024 commit
# Built using PyTorch 2.3.0 with CUDA 12.1
# All versions discovered via WebFetch from PyPI for July 2024 era

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Set environment variables for build
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV SGLANG_COMMIT=5d264a90ac5154d8e368ee558337dd3dd92e720b

# Create working directory
WORKDIR /sgl-workspace

# Create constraints file with discovered versions from PyPI (July 2024 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions discovered from PyPI for 2024-07-14 era
# FastAPI 0.111.1 was released on July 14, 2024 (exact same day)
fastapi==0.111.1
# Uvicorn 0.30.1 was released on June 2, 2024
uvicorn==0.30.1
# Pydantic 2.8.2 was released on July 4, 2024
pydantic==2.8.2
# typing_extensions 4.12.2 was released on June 7, 2024 (safe - no Sentinel)
typing_extensions==4.12.2
# Outlines 0.0.44 was released on June 14, 2024
outlines==0.0.44
# PyZMQ 26.0.3 was released on May 1, 2024
pyzmq==26.0.3
# Aiohttp 3.9.5 was the latest before July 14 (3.10.0 came July 30)
aiohttp==3.9.5
# Uvloop 0.19.0 was released Oct 2023 (0.20.0 came Aug 15, 2024)
uvloop==0.19.0
# Additional constraints from vLLM 0.5.1 requirements
numpy<2.0.0
transformers>=4.42.0
tokenizers>=0.19.1
prometheus_client>=0.18.0
prometheus-fastapi-instrumentator>=7.0.0
tiktoken>=0.6.0
lm-format-enforcer==0.10.1
filelock>=3.10.4
EOF

# Install vLLM 0.5.1 without dependencies first
RUN pip install vllm==0.5.1 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    numpy \
    requests \
    tqdm \
    py-cpuinfo \
    transformers \
    tokenizers \
    fastapi \
    aiohttp \
    openai \
    "uvicorn[standard]" \
    pydantic \
    pillow \
    prometheus_client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing_extensions \
    filelock \
    ray>=2.9 \
    nvidia-ml-py \
    torchvision==0.18.0 \
    xformers==0.0.26.post1 \
    vllm-flash-attn==2.5.9

# Install flashinfer from the official wheel repository
# Using the latest available version for cu121/torch2.3
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# Clone SGLang at the specific commit (occurrence 2 of 3)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 5d264a90ac5154d8e368ee558337dd3dd92e720b

# Verify we have the correct commit and save it (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="5d264a90ac5154d8e368ee558337dd3dd92e720b" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL" && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Build and install sgl-kernel from source
# (sgl-kernel not available on PyPI until April 2025)
RUN cd /sgl-workspace/sglang/python/sglang/srt/srt_kernels && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl

# Install SGLang without dependencies (using constraints)
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
# These are from the pyproject.toml [srt] optional deps
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
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
    pyzmq \
    openai \
    tiktoken

# Install triton-nightly (as in original Dockerfile)
RUN pip uninstall -y triton triton-nightly && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python -c "import sglang; print('SGLang imported successfully')" && \
    python -c "import vllm; print('vLLM imported successfully')" && \
    python -c "import outlines; print('Outlines imported successfully')" && \
    python -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

# Set default command
CMD ["/bin/bash"]