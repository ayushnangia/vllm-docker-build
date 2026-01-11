# Fixed Dockerfile for SGLang commit 05b3bf5e8e4751cf51510198ae2e864c4b11ac2f
# Date: 2024-10-23
# SGLang version: 0.3.4.post1
# vLLM version: 0.6.3.post1
# torch version: 2.4.0

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo libibverbs-dev \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
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

# Verify Python installation
RUN python3 --version && python3 -m pip --version

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set TORCH_CUDA_ARCH_LIST for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Pre-install torch 2.4.0 with CUDA 12.1 (MUST be done first)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.6.3.post1 with --no-deps to prevent pulling wrong torch version
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies manually (excluding torch which we already installed)
RUN pip3 install \
    nvidia-ml-py \
    "ray>=2.9" \
    torchvision==0.19 \
    xformers==0.0.27.post2

# Install flashinfer from PyPI wheel for torch 2.4 + CUDA 12.1
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install common ML/AI dependencies that SGLang needs
RUN pip3 install \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    orjson \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    zmq \
    "outlines>=0.0.44" \
    modelscope \
    requests \
    tqdm \
    numpy \
    "openai>=1.0" \
    tiktoken \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Install datamodel_code_generator for MiniCPM models
RUN pip3 install datamodel_code_generator

# Set working directory
WORKDIR /sgl-workspace

# HARDCODE commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=05b3bf5e8e4751cf51510198ae2e864c4b11ac2f

# Clone SGLang and checkout EXACT commit (occurrence 2/3 - hardcoded)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 05b3bf5e8e4751cf51510198ae2e864c4b11ac2f

# VERIFY commit matches expected SHA (occurrence 3/3 - hardcoded)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="05b3bf5e8e4751cf51510198ae2e864c4b11ac2f" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove vllm since we already installed it
# (SGLang's pyproject.toml has vllm==0.6.3.post1 which we installed with --no-deps)
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml

# Install SGLang from checked-out source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python3 -c "import xformers; print(f'xformers {xformers.__version__}')"

# Clean pip cache
RUN pip3 cache purge

# Set environment back to interactive
ENV DEBIAN_FRONTEND=interactive

# Set entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]