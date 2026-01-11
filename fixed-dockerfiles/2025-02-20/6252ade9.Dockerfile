# Docker image for SGLang
# Date: 2025-02-20
# torch 2.5.1 requires CUDA 12.1+ base image

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV SGLANG_COMMIT=6252ade98571c3374d7e7df3430a2bfbddfc5eb3

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

# Create constraints file with discovered versions from PyPI
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-02-20 era
# vLLM 0.7.2 requires pydantic>=2.9, outlines==0.1.11
# outlines 0.1.11 requires pydantic>=2.0
pydantic==2.10.6
pydantic-core==2.25.5
typing_extensions==4.12.2
fastapi==0.115.0
uvicorn==0.31.1
outlines==0.1.11
pyzmq==26.0.0
numpy<2.0.0
EOF

# Install vLLM 0.7.2 without dependencies to control versions
RUN pip install vllm==0.7.2 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    'numpy<2.0.0' \
    'requests>=2.26.0' \
    tqdm \
    blake3 \
    py-cpuinfo \
    'transformers>=4.48.2' \
    'tokenizers>=0.19.1' \
    protobuf \
    'fastapi!=0.113.*,!=0.114.0,>=0.107.0' \
    aiohttp \
    'openai>=1.52.0' \
    'uvicorn[standard]' \
    'pydantic>=2.9' \
    'prometheus_client>=0.18.0' \
    pillow \
    'prometheus-fastapi-instrumentator>=7.0.0' \
    'tiktoken>=0.6.0' \
    'lm-format-enforcer<0.11,>=0.10.9' \
    'outlines==0.1.11' \
    'lark==1.2.2' \
    'xgrammar>=0.1.6' \
    'typing_extensions>=4.10' \
    'filelock>=3.16.1' \
    partial-json-parser \
    pyzmq \
    msgspec \
    'gguf==0.10.0'

# Install torchaudio and torchvision for vLLM
RUN pip install torchaudio==2.5.1 torchvision==0.20.1

# Install flashinfer-python with correct torch/cuda version
RUN pip install flashinfer-python>=0.2.1.post2 \
    --find-links https://flashinfer.ai/whl/cu121/torch2.5/flashinfer-python

# Install sgl-kernel
RUN pip install sgl-kernel==0.1.0

# Clone SGLang repository at the exact commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 6252ade98571c3374d7e7df3430a2bfbddfc5eb3

# Verify commit SHA (3rd occurrence as required)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6252ade98571c3374d7e7df3430a2bfbddfc5eb3" && \
    test "$ACTUAL" = "$EXPECTED" || exit 1 && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    orjson \
    packaging \
    pillow \
    'prometheus-client>=0.20.0' \
    psutil \
    pydantic \
    python-multipart \
    'pyzmq>=25.1.2' \
    'torchao>=0.7.0' \
    uvicorn \
    uvloop \
    'xgrammar==0.1.10' \
    ninja \
    'transformers==4.48.3'

# Install SGLang base dependencies
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Install additional dependencies for MiniCPM models
RUN pip install datamodel_code_generator

# Verify installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

# Sanity check: Verify commit SHA matches
RUN test "$(cat /opt/sglang_commit.txt)" = "$SGLANG_COMMIT" || \
    (echo "Commit verification failed!" && exit 1)

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]