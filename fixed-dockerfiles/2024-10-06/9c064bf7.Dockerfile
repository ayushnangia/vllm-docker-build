# SGLang Dockerfile for commit 9c064bf78af8558dbc50fbd809f65dcafd6fd965 (2024-10-06)
# vLLM: 0.5.5 (from pyproject.toml)
# Base image: CUDA 12.1 for torch 2.4.0 (required by vLLM 0.5.5)
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo \
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

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 (required by vLLM 0.5.5)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies manually (excluding torch which is already installed)
# Based on vLLM 0.5.5's requirements-cuda.txt and requirements-common.txt
RUN pip3 install \
    "nvidia-ml-py" \
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
    "ray>=2.9" \
    "torchvision==0.19" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1"

# Install flashinfer from the wheel index (available for cu121/torch2.4)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Additional dependencies for SGLang (from pyproject.toml)
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
    "outlines>=0.0.44" \
    "modelscope" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0" \
    "datamodel_code_generator"  # For openbmb/MiniCPM models

# HARDCODE the commit SHA (no ARG, directly embedded in Dockerfile)
ENV SGLANG_COMMIT=9c064bf78af8558dbc50fbd809f65dcafd6fd965

# Clone SGLang and checkout EXACT commit (hardcoded SHA, not variable)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 9c064bf78af8558dbc50fbd809f65dcafd6fd965

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9c064bf78af8558dbc50fbd809f65dcafd6fd965" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed packages
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source
# Note: pyproject.toml is in python/ subdirectory for this era
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace system Triton with nightly version for better performance
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import torch; print(f'torch version: {torch.__version__}')" && \
    python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')"

# Cache cleanup
RUN pip3 cache purge

# Set environment back to interactive
ENV DEBIAN_FRONTEND=interactive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Set the entrypoint
WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]