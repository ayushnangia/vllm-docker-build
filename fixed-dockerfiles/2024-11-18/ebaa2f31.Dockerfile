# Fixed Dockerfile for SGLang commit ebaa2f31996e80e4128b832d70f29f288b59944e
# Date: 2024-11-18
# SGLang version: 0.3.5.post2
# vLLM: 0.6.3.post1 (requires torch 2.4.0 exactly)
# Target: H100 GPU

# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        sudo \
        libibverbs-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    cd .. && \
    rm -rf Python-3.10.14*

# Upgrade pip and essential packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 (required by vLLM 0.6.3.post1 exactly)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4.0 (as required by vLLM)
RUN pip3 install xformers==0.0.27.post2

# Install flashinfer for torch 2.4 + CUDA 12.1 (vLLM may need it)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.6.3.post1 with --no-deps to avoid version conflicts
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies (from requirements-common.txt and requirements-cuda.txt)
RUN pip3 install \
    "transformers>=4.45.2" \
    "tokenizers>=0.19.1" \
    "fastapi>=0.107.0" \
    "pydantic>=2.9" \
    "openai>=1.40.0" \
    "lm-format-enforcer==0.10.6" \
    "gguf==0.10.0" \
    "compressed-tensors==0.10.0" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "tiktoken>=0.6.0" \
    "outlines>=0.0.43,<0.1" \
    "mistral_common[opencv]>=1.4.4" \
    "einops" \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "uvicorn[standard]" \
    "prometheus-client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "pyzmq" \
    "msgspec" \
    "importlib_metadata" \
    "jinja2" \
    "lark" \
    "pillow" \
    "triton>=2.0.0" \
    "pynvml" \
    "psutil"

# Additional dependencies for SGLang
RUN pip3 install \
    aiohttp \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    python-multipart \
    torchao \
    uvloop \
    IPython \
    tqdm \
    requests \
    html5lib \
    six

# For openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator

# HARDCODE the SGLang commit SHA (OCCURRENCE 1/3)
ENV SGLANG_COMMIT=ebaa2f31996e80e4128b832d70f29f288b59944e

# Clone SGLang repo and checkout EXACT commit (OCCURRENCE 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ebaa2f31996e80e4128b832d70f29f288b59944e

# VERIFY the checkout - compare against HARDCODED expected value (OCCURRENCE 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ebaa2f31996e80e4128b832d70f29f288b59944e" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
ENV TORCH_CUDA_ARCH_LIST="9.0"
RUN pip3 install -e "python[all]"

# Replace Triton with nightly version if needed
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify imports
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Clean up pip cache
RUN python3 -m pip cache purge

ENV DEBIAN_FRONTEND=interactive

# Set working directory for runtime
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]