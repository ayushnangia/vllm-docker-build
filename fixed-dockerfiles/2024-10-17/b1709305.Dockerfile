# Dockerfile for SGLang commit (2024-10-17)
# vLLM 0.5.5 requires torch 2.4.0, using CUDA 12.1
# Commit SHA is hardcoded exactly 3 times below for verification

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

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

# HARDCODE the commit SHA (not using ARG to avoid forgotten --build-arg issues)
ENV SGLANG_COMMIT=b170930534acbb9c1619a3c83670a839ceee763a

# Pre-install torch 2.4.0 with CUDA 12.1 index
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 without dependencies to avoid version conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies (from vllm 0.5.5 requirements-cuda.txt)
RUN pip3 install \
    "ray>=2.9" \
    "nvidia-ml-py" \
    "torchvision==0.19" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1"

# Install vLLM common requirements (from vllm 0.5.5 requirements-common.txt)
RUN pip3 install \
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
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "pyzmq" \
    "msgspec" \
    "librosa" \
    "soundfile" \
    "gguf==0.9.1" \
    "importlib_metadata"

# Install flashinfer using prebuilt wheel for torch 2.4.0 + CUDA 12.1
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang and checkout EXACT commit (hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout b170930534acbb9c1619a3c83670a839ceee763a

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b170930534acbb9c1619a3c83670a839ceee763a" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove flashinfer (already installed)
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# For openbmb/MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# Clear pip cache
RUN pip3 cache purge

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]