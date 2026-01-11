# SGLang Dockerfile for commit 5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9 (2024-10-17)
# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# HARDCODED COMMIT SHA (1st occurrence) - ENV variable
ENV SGLANG_COMMIT=5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9

# Set up timezone
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections

# Install system dependencies for Python build
RUN apt-get update -y && \
    apt-get install -y \
        git curl wget \
        build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
        libibverbs-dev \
        cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA deprecated on Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    cd .. && rm -rf Python-3.10.14*

# Verify Python version
RUN python3 --version && python3 -m pip --version

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Set working directory
WORKDIR /sgl-workspace

# Create constraints file with discovered versions from PyPI for 2024-10-17 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for October 17, 2024
# CRITICAL: outlines 0.0.44 requires pydantic >=2.0 (discovered via metadata)
pydantic==2.9.2
pydantic-core==2.23.4
fastapi==0.115.2
uvicorn==0.32.0
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
EOF

# CRITICAL: Install torch 2.4.0 FIRST with CUDA 12.1 (required by vLLM 0.5.5)
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install numpy < 2.0.0 before other packages to avoid conflicts
RUN pip3 install "numpy<1.27.0"

# Install vLLM 0.5.5 WITHOUT dependencies to avoid torch conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies from requirements-common.txt and requirements-cuda.txt
# Use constraints to ensure compatibility with October 2024 packages
RUN pip3 install -c /opt/constraints.txt \
    "psutil" \
    "sentencepiece" \
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
    "nvidia-ml-py" \
    "torchvision==0.19.0"

# Install xformers 0.0.27.post2 for torch 2.4.0 compatibility
RUN pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install vllm-flash-attn
RUN pip3 install vllm-flash-attn==2.6.1

# Install flashinfer from wheels (available for torch 2.4 + CUDA 12.1)
# Version 0.2.0.post2 discovered for October 2024 era
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install sgl-kernel from PyPI
# Version 0.3.16.post2 was released Oct 14, 2024 (discovered via PyPI history)
RUN pip3 install sgl-kernel==0.3.16.post2

# Clone and checkout the specific commit (HARDCODED SHA - 2nd occurrence)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9

# Verify we have the correct commit (HARDCODED SHA - 3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: got $ACTUAL, expected $EXPECTED" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang WITHOUT dependencies (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang runtime_common dependencies
# Based on pyproject.toml runtime_common section discovered at this commit
RUN pip3 install -c /opt/constraints.txt \
    "aiohttp" \
    "decord" \
    "fastapi" \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "orjson" \
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
    "modelscope"

# Install SGLang optional dependencies for "all" extra
RUN pip3 install -c /opt/constraints.txt \
    "openai>=1.0" \
    "tiktoken" \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# For MiniCPM models
RUN pip3 install datamodel_code_generator

# Clear pip cache
RUN python3 -m pip cache purge

# Final verification that everything works
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')"

# Reset environment
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace/sglang

# Set default command
CMD ["/bin/bash"]