# Dockerfile for SGLang commit ddcf9fe3beacd8aed573c711942194dd02350da4
# Date: 2025-02-20
# Based on discovered dependencies:
# - torch 2.5.1 (from vLLM 0.6.4 requirements)
# - vLLM 0.6.4.post1
# - flashinfer_python 0.2.5 (wheel available)
# - sgl-kernel 0.3.18 (from PyPI)
# - xformers 0.0.28.post3

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.1
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# First occurrence of SHA: Environment variable
ENV SGLANG_COMMIT=ddcf9fe3beacd8aed573c711942194dd02350da4

# Install system dependencies and Python 3.10
RUN apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        wget \
        curl \
        git \
        vim \
        build-essential \
        cmake \
        ninja-build \
        libibverbs-dev \
        rdma-core \
        infiniband-diags \
        openssh-server \
        perftest \
        ibverbs-providers \
        libibumad3 \
        libibverbs1 \
        libnl-3-200 \
        libnl-route-3-200 \
        librdmacm1 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify Python version
RUN python3 --version && python3 -m pip --version

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.5.1 with CUDA 12.1
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install xformers 0.0.28.post3 for torch 2.5.1
RUN pip install --no-cache-dir xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.6.4.post1 without dependencies to avoid pulling wrong torch
RUN pip install --no-cache-dir vllm==0.6.4.post1 --no-deps

# Install vLLM dependencies (from requirements-common.txt and requirements-cuda.txt)
RUN pip install --no-cache-dir \
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
    "openai>=1.45.0" \
    "uvicorn[standard]" \
    "pydantic>=2.9" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer>=0.10.9,<0.11" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "partial-json-parser" \
    "pyzmq" \
    "msgspec" \
    "gguf==0.10.0" \
    "importlib_metadata" \
    "mistral_common[opencv]>=1.5.0" \
    "pyyaml" \
    "einops" \
    "compressed-tensors==0.8.0" \
    "nvidia-ml-py>=12.560.30" \
    "ray>=2.9" \
    "torchvision==0.20.1"

# Install flashinfer_python 0.2.5 from wheel
RUN pip install --no-cache-dir flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu121/torch2.5/flashinfer-python/

# Install sgl-kernel from PyPI (using 0.3.18)
RUN pip install --no-cache-dir sgl-kernel==0.3.18

# Second occurrence of SHA: Clone and checkout SGLang at specific commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout ddcf9fe3beacd8aed573c711942194dd02350da4

# Third occurrence of SHA: Verify commit and write to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ddcf9fe3beacd8aed573c711942194dd02350da4" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    "cuda-python" \
    "decord" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "modelscope" \
    "orjson" \
    "packaging" \
    "python-multipart" \
    "torchao>=0.7.0" \
    "xgrammar==0.1.10" \
    "ninja" \
    "transformers==4.48.3" \
    "IPython" \
    "setproctitle"

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip install --no-cache-dir -e "python[all]"

# Install additional OpenAI/Anthropic/LiteLLM support
RUN pip install --no-cache-dir \
    "openai>=1.0" \
    "tiktoken" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import xformers; print('xformers import OK')"

# Set environment to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

# Set entrypoint
WORKDIR /sgl-workspace