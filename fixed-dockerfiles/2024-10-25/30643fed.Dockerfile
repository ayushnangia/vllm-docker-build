# SGLang Dockerfile for commit 30643fed7f92be32540dfcdf9e4310e477ce0f6d (2024-10-25)
# SGLang version: 0.3.4.post2
# vLLM: 0.6.3.post1, torch: 2.4.0
# Using nvidia/cuda base for torch 2.4.0
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Build settings
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# HARDCODED SHA (1 of 3)
ENV SGLANG_COMMIT=30643fed7f92be32540dfcdf9e4310e477ce0f6d

# Install system dependencies
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
RUN python3 --version && pip3 --version

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 (required by vLLM 0.6.3.post1)
RUN pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers for torch 2.4.0
RUN pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions for October 25, 2024
# All versions were discovered via WebFetch from PyPI release histories
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-10-25 era
# fastapi 0.115.3 released Oct 22, 2024 (discovered via WebFetch)
fastapi==0.115.3
# uvicorn 0.32.0 released Oct 15, 2024 (discovered via WebFetch)
uvicorn==0.32.0
# pydantic 2.9.2 released Sep 17, 2024 (discovered via WebFetch)
# outlines 0.0.44 requires pydantic>=2.0 (discovered via WebFetch)
pydantic==2.9.2
# typing_extensions 4.12.2 released Jun 7, 2024 (discovered via WebFetch)
typing_extensions==4.12.2
# outlines 0.0.44 from pyproject.toml requirement
outlines==0.0.44
# pyzmq 26.2.0 released Aug 22, 2024 (discovered via WebFetch)
pyzmq==26.2.0
EOF

# Install vLLM 0.6.3.post1 with --no-deps first
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with version constraints where critical
RUN pip3 install -c /opt/constraints.txt \
    psutil sentencepiece 'numpy<2.0.0' 'requests>=2.26.0' tqdm py-cpuinfo \
    'transformers>=4.45.2' 'tokenizers>=0.19.1' protobuf \
    'fastapi>=0.107.0,!=0.113.*,!=0.114.0' aiohttp 'openai>=1.40.0' \
    'uvicorn[standard]' 'pydantic>=2.9' pillow 'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' 'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.6' 'outlines>=0.0.43,<0.1' 'typing_extensions>=4.10' \
    'filelock>=3.10.4' partial-json-parser pyzmq msgspec 'gguf==0.10.0' \
    importlib_metadata 'mistral_common[opencv]>=1.4.4' pyyaml 'six>=1.16.0' \
    'setuptools>=74.1.1' einops 'compressed-tensors==0.6.0' \
    ray 'nvidia-ml-py' vllm-nccl-cu12

# Install flashinfer from wheels (available for cu121/torch2.4)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install sgl-kernel from PyPI (version 0.3.4.post1 discovered via WebFetch)
RUN pip3 install sgl-kernel==0.3.4.post1

# Additional dependencies for compatibility
RUN pip3 install datamodel_code_generator

# Create workspace
WORKDIR /sgl-workspace

# Clone SGLang at specific commit (HARDCODED SHA 2 of 3)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 30643fed7f92be32540dfcdf9e4310e477ce0f6d

# Verify commit SHA (HARDCODED SHA 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="30643fed7f92be32540dfcdf9e4310e477ce0f6d" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL: COMMIT MISMATCH!" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps first
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang runtime dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    aiohttp decord fastapi hf_transfer huggingface_hub interegular \
    orjson packaging pillow psutil pydantic python-multipart \
    torchao uvicorn uvloop zmq 'outlines>=0.0.44' modelscope \
    'openai>=1.0' tiktoken 'anthropic>=0.20.0' 'litellm>=1.0.0'

# Clean pip cache
RUN pip3 cache purge

# Verify installations
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')" && \
    python3 -c "import outlines; print('Outlines import OK')"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists with content:" && \
    cat /opt/sglang_commit.txt

# Set environment for runtime
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["/bin/bash"]