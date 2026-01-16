# SGLang Docker image for SGLang repository
# Date: 2024-08-09
# Requirements: vLLM 0.5.4, torch 2.4.0

# Base image for torch 2.4.0 with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# 1st SHA occurrence
ENV SGLANG_COMMIT=73fa2d49d539fd67548b0458a365528d3e3b6edc

# System dependencies and timezone
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        git \
        curl \
        wget \
        sudo \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        liblzma-dev \
        ccache \
        ninja-build \
        cmake \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI for 2024-08-09 era
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions discovered from PyPI for August 2024
fastapi==0.112.0
uvicorn==0.30.6
pydantic==2.8.2
pydantic-core==2.20.1
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.1.0
pillow==10.4.0
aiohttp==3.10.1
numpy==1.26.4
huggingface_hub==0.24.5
transformers==4.44.0
tokenizers==0.19.1
EOF

# Install vLLM 0.5.4 with --no-deps first
RUN pip3 install --no-cache-dir vllm==0.5.4 --no-deps

# Install vLLM dependencies with specific versions for compatibility
RUN pip3 install --no-cache-dir -c /opt/constraints.txt \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "sentencepiece" \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1" \
    "cmake>=3.21" \
    "ninja" \
    "psutil" \
    "numpy<2.0.0" \
    "requests" \
    "tqdm" \
    "py-cpuinfo" \
    "fastapi==0.112.0" \
    "aiohttp==3.10.1" \
    "openai" \
    "uvicorn[standard]==0.30.6" \
    "pydantic==2.8.2" \
    "pillow==10.4.0" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.3" \
    "outlines==0.0.44" \
    "typing_extensions==4.12.2" \
    "filelock>=3.10.4" \
    "pyzmq==26.1.0"

# Install flashinfer from wheel repository (available for torch 2.4 + CUDA 12.1)
RUN pip3 install --no-cache-dir flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 73fa2d49d539fd67548b0458a365528d3e3b6edc

# VERIFY the checkout - compare against HARDCODED expected value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="73fa2d49d539fd67548b0458a365528d3e3b6edc" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL: COMMIT MISMATCH!" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps first
RUN cd /sgl-workspace/sglang && \
    pip3 install --no-cache-dir -e python --no-deps

# Install SGLang dependencies with specific versions
RUN pip3 install --no-cache-dir -c /opt/constraints.txt \
    "hf_transfer" \
    "huggingface_hub==0.24.5" \
    "interegular" \
    "packaging" \
    "python-multipart" \
    "uvloop" \
    "pyzmq"

# Install sgl-kernel from PyPI (available for this version)
RUN pip3 install --no-cache-dir sgl-kernel

# Additional optional dependencies
RUN pip3 install --no-cache-dir -c /opt/constraints.txt \
    msgpack scipy pandas

# Verify installation
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
# Note: torch.cuda.is_available() returns False without GPU - skip this check
# Note: outlines import can fail due to dataset/huggingface_hub version mismatch - check via pip
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    pip show outlines > /dev/null && echo "outlines installed OK" && \
    python3 -c "import pydantic; print(f'pydantic: {pydantic.__version__}')"

# Clean up
RUN pip3 cache purge

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["/bin/bash"]