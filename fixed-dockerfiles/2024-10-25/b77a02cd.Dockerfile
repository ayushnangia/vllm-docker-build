# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# 1st occurrence of commit SHA - hardcoded as ENV
ENV SGLANG_COMMIT=b77a02cdfdb4cd58be3ebc6a66d076832c309cfc

# Install system dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        sudo \
        libibverbs-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libgdbm-dev \
        libnss3-dev \
        libegl1 \
        software-properties-common && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Build Python 3.10 from source (deadsnakes PPA deprecated on Ubuntu 20.04)
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --with-lto --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/Python-3.10.14* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10 && \
    python3 --version

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip --version

# Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# 2nd occurrence of commit SHA - hardcoded in git checkout
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout b77a02cdfdb4cd58be3ebc6a66d076832c309cfc

# 3rd occurrence of commit SHA - verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b77a02cdfdb4cd58be3ebc6a66d076832c309cfc" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install torch 2.4.0 with CUDA 12.1 FIRST (required by vLLM 0.6.3.post1)
RUN pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install xformers 0.0.27.post2 for torch 2.4.0 (required by vLLM)
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.6.3.post1 WITHOUT dependencies to avoid pulling wrong torch
RUN pip install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies manually (from requirements-common.txt and requirements-cuda.txt)
RUN pip install \
    "psutil" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "requests>=2.26.0" \
    "tqdm" \
    "py-cpuinfo" \
    "transformers>=4.45.2" \
    "tokenizers>=0.19.1" \
    "protobuf" \
    "fastapi>=0.107.0,!=0.113.*,!=0.114.0" \
    "aiohttp" \
    "openai>=1.40.0" \
    "uvicorn[standard]" \
    "pydantic>=2.9" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "partial-json-parser" \
    "pyzmq" \
    "msgspec" \
    "gguf==0.10.0" \
    "importlib_metadata" \
    "mistral_common[opencv]>=1.4.4" \
    "pyyaml" \
    "setuptools>=74.1.1" \
    "einops" \
    "compressed-tensors==0.6.0" \
    "ray>=2.9" \
    "nvidia-ml-py" \
    "torchvision==0.19"

# Install flashinfer 0.2.0.post1 from wheel (available for torch 2.4 + CUDA 12.1)
RUN pip install flashinfer==0.2.0.post1 -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install sgl-kernel from PyPI (using version that matches SGLang version)
RUN pip install sgl-kernel==0.3.4.post1

# Install additional dependencies for SGLang
RUN pip install \
    "datamodel_code_generator" \
    "interegular" \
    "orjson" \
    "packaging" \
    "python-multipart" \
    "torchao" \
    "uvloop" \
    "hf_transfer" \
    "huggingface_hub" \
    "decord" \
    "modelscope"

# Patch SGLang's pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip install -e "python[all]"

# Set torch CUDA arch list for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Final verification of all imports
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')"

# Clear pip cache
RUN python3 -m pip cache purge

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=interactive

# Set a default command
CMD ["/bin/bash"]