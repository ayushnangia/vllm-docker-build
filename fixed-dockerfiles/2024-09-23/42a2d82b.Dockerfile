# Fixed Dockerfile for SGLang commit 42a2d82ba71dc86ca3b6342c978db450658b750c
# Date: 2024-09-23
# vLLM 0.5.5 requires torch 2.4.x
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo \
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
RUN python3 -m pip install --upgrade pip setuptools wheel

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=42a2d82ba71dc86ca3b6342c978db450658b750c

# Pre-install torch 2.4.0 with CUDA 12.1
RUN python3 -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.4
RUN python3 -m pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from PyPI wheel (available for torch 2.4 + CUDA 12.1)
RUN python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.5.5 with --no-deps to avoid dependency conflicts
RUN python3 -m pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies manually (minus torch which is already installed)
RUN python3 -m pip install \
    "nvidia-ml-py" \
    "sentencepiece" \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    "protobuf" \
    "fastapi" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "aioprometheus" \
    "pynvml==11.5.0" \
    "triton>=2.2.0" \
    "py-cpuinfo" \
    "typing-extensions>=4.10" \
    "filelock>=3.10.4" \
    "pyzmq" \
    "ray>=2.9" \
    "tensorizer>=2.9.0" \
    "msgspec" \
    "gguf==0.9.1"

# Install additional SGLang dependencies
RUN python3 -m pip install \
    "aiohttp" \
    "decord" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "packaging" \
    "pillow" \
    "psutil" \
    "python-multipart" \
    "torchao" \
    "uvloop" \
    "zmq" \
    "outlines>=0.0.44" \
    "openai>=1.0" \
    "tiktoken" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0" \
    "requests" \
    "tqdm" \
    "numpy" \
    "datamodel_code_generator"

# Clone SGLang repo and checkout EXACT commit (HARDCODED - 2nd occurrence)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 42a2d82ba71dc86ca3b6342c978db450658b750c

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="42a2d82ba71dc86ca3b6342c978db450658b750c" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove flashinfer dependency (already installed)
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN python3 -m pip install --no-cache-dir -e "python[all]"

# Replace Triton with nightly version if needed
RUN python3 -m pip uninstall -y triton triton-nightly || true && \
    python3 -m pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Clear pip cache
RUN python3 -m pip cache purge

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Set environment
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]