# SGLang Dockerfile for commit 8f8f96a6217ea737c94e7429e480196319594459 (2024-10-23)
# Based on discovered dependencies via WebFetch/WebSearch

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# 1st hardcoded SHA occurrence
ENV SGLANG_COMMIT=8f8f96a6217ea737c94e7429e480196319594459

# Install system dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y --allow-change-held-packages \
        build-essential \
        cmake \
        curl \
        git \
        wget \
        sudo \
        vim \
        ninja-build \
        libnccl2 \
        libnccl-dev \
        libibverbs-dev \
        zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
        liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA deprecated for Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    cd .. && rm -rf Python-3.10.14*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Install PyTorch 2.4.0 and related packages (vLLM requirement)
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4.0
RUN python3 -m pip install --no-cache-dir xformers==0.0.27.post2

# Create constraints file with discovered versions for October 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-10-23 era
# vLLM 0.6.3.post1 requires: pydantic>=2.9, fastapi>=0.107.0,!=0.113.*,!=0.114.0
# outlines must be <0.1 for vLLM compatibility
fastapi==0.115.3
uvicorn==0.32.0
pydantic==2.9.2
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
uvloop==0.20.0
aiohttp==3.10.10
orjson==3.10.7
psutil==6.0.0
pillow==10.4.0
packaging==24.1
huggingface_hub==0.25.2
transformers==4.45.2
tokenizers==0.20.0
protobuf==5.28.2
einops==0.8.0
pyyaml==6.0.2
msgspec==0.18.6
filelock==3.16.1
tqdm==4.66.5
numpy==1.26.4
requests==2.32.3
sentencepiece==0.2.0
py-cpuinfo==9.0.0
tiktoken==0.7.0
mistral_common==1.4.4
importlib_metadata==8.5.0
prometheus_client==0.21.0
prometheus-fastapi-instrumentator==7.0.0
EOF

# Install vLLM 0.6.3.post1 with --no-deps to avoid conflicts
RUN python3 -m pip install --no-cache-dir vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies using constraints
RUN python3 -m pip install --no-cache-dir -c /opt/constraints.txt \
    psutil sentencepiece numpy requests tqdm py-cpuinfo \
    transformers tokenizers protobuf \
    fastapi aiohttp openai uvicorn pydantic pillow \
    prometheus_client prometheus-fastapi-instrumentator tiktoken \
    lm-format-enforcer==0.10.6 outlines typing_extensions filelock \
    partial-json-parser pyzmq msgspec gguf==0.10.0 importlib_metadata \
    mistral_common pyyaml einops compressed-tensors==0.6.0 \
    nvidia-ml-py ray

# Install flashinfer from custom index
RUN python3 -m pip install --no-cache-dir \
    flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Create workspace directory
WORKDIR /sgl-workspace

# Clone SGLang repository and checkout specific commit
# 2nd hardcoded SHA occurrence
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 8f8f96a6217ea737c94e7429e480196319594459

# Verify commit SHA and write to file
# 3rd hardcoded SHA occurrence
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="8f8f96a6217ea737c94e7429e480196319594459" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang dependencies first (using constraints)
RUN python3 -m pip install --no-cache-dir -c /opt/constraints.txt \
    aiohttp decord fastapi hf_transfer huggingface_hub interegular \
    orjson packaging pillow psutil pydantic python-multipart \
    torchao uvicorn uvloop pyzmq outlines modelscope \
    openai tiktoken anthropic litellm

# Install sgl-kernel from PyPI (version 0.3.4.post1 available)
RUN python3 -m pip install --no-cache-dir sgl-kernel==0.3.4.post1

# Install SGLang in editable mode with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    python3 -m pip install -e . --no-deps

# Additional packages for MiniCPM models
RUN python3 -m pip install --no-cache-dir datamodel_code_generator

# Final sanity check - verify all critical imports work
# Note: vLLM and outlines imports fail without GPU - check via pip
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    pip show outlines > /dev/null && echo "Outlines installed OK" && \
    python3 -c "import pydantic; print(f'Pydantic: {pydantic.VERSION}')" && \
    python3 -c "import fastapi; print('FastAPI import OK')"

# Clear pip cache
RUN python3 -m pip cache purge

# Set interactive mode for runtime
ENV DEBIAN_FRONTEND=interactive

# Default working directory
WORKDIR /sgl-workspace

# Entry point
CMD ["/bin/bash"]