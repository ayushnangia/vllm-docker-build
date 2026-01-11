# Fixed Dockerfile for SGLang
# Date: 2025-02-20
# Based on exploration of actual repo requirements

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=6252ade98571c3374d7e7df3430a2bfbddfc5eb3
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Install system dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        curl \
        git \
        wget \
        vim \
        build-essential \
        cmake \
        ninja-build \
        libibverbs-dev \
        rdma-core \
        infiniband-diags \
        ibverbs-providers \
        libibumad3 \
        libibverbs1 \
        libnl-3-200 \
        libnl-route-3-200 \
        librdmacm1 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.5.1 with CUDA 12.4 (required by vLLM 0.6.4)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install xformers 0.0.28.post3 (required by vLLM 0.6.4)
RUN pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124

# Install vLLM without dependencies to control versions
RUN pip install vllm==0.6.4.post1 --no-deps

# Install vLLM dependencies manually (from vLLM requirements-cuda.txt and requirements-common.txt)
RUN pip install \
    "ray>=2.9" \
    "nvidia-ml-py>=12.560.30" \
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
    "six>=1.16.0" \
    "setuptools>=74.1.1" \
    "einops" \
    "compressed-tensors==0.8.0"

# Install flashinfer from wheels (version >= 0.2.1.post2 as required)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python/

# Install sgl-kernel from PyPI (version >= 0.0.3.post6 as required)
RUN pip install sgl-kernel

# Clone SGLang repository at the exact commit
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 6252ade98571c3374d7e7df3430a2bfbddfc5eb3

# Verify commit SHA (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6252ade98571c3374d7e7df3430a2bfbddfc5eb3" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: got $ACTUAL, expected $EXPECTED" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install additional dependencies for MiniCPM models
RUN pip install datamodel_code_generator

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip install -e "python[all]"

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import vllm; print('vLLM OK')" && \
    python3 -c "import xformers; print('xformers OK')"

# Set environment back to interactive
ENV DEBIAN_FRONTEND=interactive

# Default command
CMD ["/bin/bash"]