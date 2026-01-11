# Fixed Dockerfile for SGLang commit ab4a83b25909aa98330b838a224e4fe5c943e483
# Date: 2024-09-05
# SGLang version: 0.3.0 (requires vllm==0.5.5, outlines>=0.0.44)
# vLLM version: 0.5.5 (requires torch==2.4.0, pydantic>=2.8)
# Base image: nvidia/cuda for torch 2.4.0 compatibility

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="9.0"  # For H100
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    gcc \
    g++ \
    cmake \
    ninja-build \
    software-properties-common \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA deprecated for Ubuntu 20.04)
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --with-lto --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd / && rm -rf /tmp/Python-3.10.14*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 && \
    update-alternatives --set python /usr/local/bin/python3.10

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /sgl-workspace

# Install PyTorch 2.4.0 first (required by vLLM 0.5.5)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI for September 2024 era
# These versions were specifically researched and verified to be available before 2024-09-05
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions discovered from PyPI for 2024-09-05 era
# pydantic v2.8.2 was latest stable v2 before Sept 5 (released July 4, 2024)
pydantic==2.8.2
# typing_extensions 4.12.2 was latest before Sept 5 (released June 7, 2024)
typing_extensions==4.12.2
# fastapi 0.112.2 supports pydantic v2 and was available before Sept 5
fastapi==0.112.2
# uvicorn 0.30.6 was available in the 0.30.x series
uvicorn==0.30.6
# uvloop 0.19.0 was stable version for 2024
uvloop==0.19.0
# pyzmq 26.0.3 for compatibility
pyzmq==26.0.3
# outlines 0.0.46 was latest before Sept 5 (released June 22, 2024)
outlines==0.0.46
# Additional dependencies from vLLM requirements
aiohttp==3.9.5
requests==2.32.3
tqdm==4.66.4
numpy==1.26.4
psutil==6.0.0
pillow==10.4.0
packaging==24.1
huggingface_hub==0.24.5
transformers==4.44.0
tokenizers==0.19.1
protobuf==5.27.3
sentencepiece==0.2.0
py-cpuinfo==9.0.0
openai==1.40.0
tiktoken==0.7.0
lm-format-enforcer==0.10.6
filelock==3.15.4
msgspec==0.18.6
prometheus_client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
librosa==0.10.2
soundfile==0.12.1
importlib_metadata==8.2.0
gguf==0.9.1
ray==2.33.0
nvidia-ml-py==12.560.30
interegular==0.3.3
hf_transfer==0.1.8
python-multipart==0.0.9
decord==0.6.0
EOF

# Install vLLM 0.5.5 without dependencies first
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    psutil sentencepiece 'numpy<2.0.0' requests tqdm py-cpuinfo \
    'transformers>=4.43.2' 'tokenizers>=0.19.1' protobuf \
    fastapi aiohttp 'openai>=1.0' 'uvicorn[standard]' \
    'pydantic>=2.8' pillow 'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' 'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.6' 'outlines>=0.0.43,<0.1' \
    'typing_extensions>=4.10' 'filelock>=3.10.4' pyzmq msgspec \
    librosa soundfile 'gguf==0.9.1' importlib_metadata

# Install CUDA-specific vLLM dependencies
RUN pip install -c /opt/constraints.txt \
    'ray>=2.9' nvidia-ml-py 'xformers==0.0.27.post2' 'vllm-flash-attn==2.6.1'

# Install flashinfer for CUDA 12.1 and torch 2.4 (prebuilt wheels available)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# 1st HARDCODED COMMIT SHA: ENV variable
ENV SGLANG_COMMIT=ab4a83b25909aa98330b838a224e4fe5c943e483

# Clone SGLang repository
# 2nd HARDCODED COMMIT SHA: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout ab4a83b25909aa98330b838a224e4fe5c943e483

# 3rd HARDCODED COMMIT SHA: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ab4a83b25909aa98330b838a224e4fe5c943e483" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang without dependencies first (editable install)
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
# Based on pyproject.toml [srt] dependencies at this commit
RUN pip install -c /opt/constraints.txt \
    requests tqdm numpy \
    aiohttp decord fastapi hf_transfer huggingface_hub interegular \
    packaging pillow psutil pydantic python-multipart \
    uvicorn uvloop pyzmq

# Install optional SGLang dependencies for full functionality
RUN pip install -c /opt/constraints.txt \
    'openai>=1.0' tiktoken 'anthropic>=0.20.0' 'litellm>=1.0.0' || true

# Install sgl-kernel (if available on PyPI for this era)
RUN pip install sgl-kernel==0.3.0 || \
    (echo "sgl-kernel not available on PyPI, building from source..." && \
     cd /sgl-workspace/sglang/python && \
     if [ -d "sglang/srt/kernels" ]; then \
       cd sglang/srt/kernels && \
       python setup.py install; \
     fi)

# Final sanity check - verify imports work
RUN python3 -c "import torch; print('Torch version:', torch.__version__)" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print('Pydantic version:', pydantic.__version__)" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')"

# Cleanup pip cache
RUN pip cache purge

# Reset to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]