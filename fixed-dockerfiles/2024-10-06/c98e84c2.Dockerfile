# SGLang Dockerfile for commit c98e84c21e4313d7d307425ca43e61753a53a9f7
# Date: 2024-10-06
# SGLang version: 0.3.2
# vLLM version: 0.5.5
# torch version: 2.4.0
# CUDA: 12.1

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# HARDCODE the commit SHA (1/3)
ENV SGLANG_COMMIT=c98e84c21e4313d7d307425ca43e61753a53a9f7

# Install system dependencies and Python 3.10
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        build-essential \
        curl \
        git \
        sudo \
        libibverbs-dev \
        cmake \
        ninja-build \
        wget \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev && \
    # Build Python 3.10 from source (deadsnakes PPA deprecated on Ubuntu 20.04)
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared --with-lto && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd .. && \
    rm -rf Python-3.10.14 Python-3.10.14.tgz && \
    # Set Python 3.10 as default
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10 && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    # Install pip
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    # Clean apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify Python installation
RUN python3 --version && python3 -m pip --version

# Upgrade pip, setuptools, wheel
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /sgl-workspace

# Install torch 2.4.0 FIRST with specific CUDA 12.1 build
RUN pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4.0
RUN pip3 install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from prebuilt wheels (available for torch 2.4 + CUDA 12.1)
RUN pip3 install --no-cache-dir flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.5.5 without dependencies to control torch version
RUN pip3 install --no-cache-dir vllm==0.5.5 --no-deps

# Install vLLM dependencies (from vLLM requirements-common.txt)
RUN pip3 install --no-cache-dir \
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
    "importlib_metadata" \
    "nvidia-ml-py"

# Clone SGLang and checkout EXACT commit (2/3 - hardcoded SHA)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout c98e84c21e4313d7d307425ca43e61753a53a9f7

# VERIFY commit - compare against HARDCODED value (3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="c98e84c21e4313d7d307425ca43e61753a53a9f7" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install additional dependencies that SGLang needs
RUN pip3 install --no-cache-dir \
    "decord" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "packaging" \
    "python-multipart" \
    "uvloop" \
    "modelscope" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0" \
    "datamodel_code_generator"  # For MiniCPM models

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip3 install --no-cache-dir -e "python[all]"

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Clear pip cache
RUN pip3 cache purge

# Final verification - ensure all critical imports work
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')"

# Set interactive mode back
ENV DEBIAN_FRONTEND=interactive

# Add a note about the commit
RUN echo "SGLang commit: c98e84c21e4313d7d307425ca43e61753a53a9f7" >> /opt/sglang_info.txt && \
    echo "Build date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> /opt/sglang_info.txt

WORKDIR /sgl-workspace/sglang

# Optional entrypoint for server launch
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]