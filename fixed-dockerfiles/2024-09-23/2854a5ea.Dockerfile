# Fixed Dockerfile for SGLang commit 2854a5ea9fbb31165936f633ab99915dec760f8d
# Date: 2024-09-23
# vLLM 0.5.5 requires torch 2.4.0

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
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
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 (required by vLLM 0.5.5)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies manually with correct versions (from vLLM 0.5.5 requirements-cuda.txt)
RUN pip3 install \
    "ray>=2.9" \
    "nvidia-ml-py" \
    "torchvision==0.19" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1"

# Install flashinfer from wheel (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install common dependencies for SGLang
RUN pip3 install \
    "aiohttp" \
    "decord" \
    "fastapi" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "packaging" \
    "pillow" \
    "psutil" \
    "pydantic" \
    "python-multipart" \
    "torchao" \
    "uvicorn" \
    "uvloop" \
    "zmq" \
    "outlines>=0.0.44" \
    "requests" \
    "tqdm" \
    "numpy" \
    "openai>=1.0" \
    "tiktoken" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=2854a5ea9fbb31165936f633ab99915dec760f8d

# Clone SGLang and checkout EXACT commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 2854a5ea9fbb31165936f633ab99915dec760f8d

# VERIFY commit - compare against HARDCODED value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="2854a5ea9fbb31165936f633ab99915dec760f8d" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove pre-installed dependencies
RUN sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install additional optional dependencies for full functionality
RUN pip3 install -e "python[openai]" && \
    pip3 install -e "python[anthropic]" && \
    pip3 install -e "python[litellm]"

# Replace Triton with nightly version if needed
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# For MiniCPM models support
RUN pip3 install datamodel_code_generator

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'torch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')"

# Clear pip cache
RUN pip3 cache purge

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace

# Set default entry point for SGLang server
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]