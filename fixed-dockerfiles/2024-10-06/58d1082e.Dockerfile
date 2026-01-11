# Fixed Dockerfile for SGLang commit 58d1082e392cabbf26c404cb7ec18e4cb51b99e9
# Date: 2024-10-06
# vLLM 0.5.5 requires torch 2.4.0, CUDA 12.1

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update && apt-get install -y \
    git curl wget sudo libibverbs-dev \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 \
    && cd .. && rm -rf Python-3.10.14*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 (as required by vLLM 0.5.5)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 without dependencies to avoid torch version conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies manually (based on requirements-cuda.txt)
RUN pip3 install \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "torchvision==0.19" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1" \
    "psutil" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "requests" \
    "tqdm" \
    "py-cpuinfo" \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    "protobuf" \
    "fastapi" \
    "aiohttp" \
    "openai>=1.0" \
    "uvicorn[standard]" \
    "pydantic>=2.8" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1"

# Install flashinfer from PyPI wheels (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=58d1082e392cabbf26c404cb7ec18e4cb51b99e9

# Clone SGLang and checkout EXACT commit (2nd occurrence of SHA, hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 58d1082e392cabbf26c404cb7ec18e4cb51b99e9

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="58d1082e392cabbf26c404cb7ec18e4cb51b99e9" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install additional dependencies for SGLang
RUN pip3 install \
    "decord" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "packaging" \
    "python-multipart" \
    "torchao" \
    "uvloop" \
    "zmq" \
    "modelscope"

# Install SGLang from checked-out source
# Note: pyproject.toml is in python/ subdirectory
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# For openbmb/MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# Clear pip cache
RUN python3 -m pip cache purge

# Verification step - ensure SGLang imports correctly
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')"

# Set environment for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Set working directory
WORKDIR /sgl-workspace

ENV DEBIAN_FRONTEND=interactive

# Add entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]