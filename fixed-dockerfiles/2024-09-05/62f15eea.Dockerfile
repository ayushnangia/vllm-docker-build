# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=62f15eea5a0b4266cdae965d0337fd33f6673736

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo vim \
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

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.4.0 with CUDA 12.1 index (required by vLLM 0.5.5)
RUN pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.5.5 --no-deps

# Install vLLM dependencies manually (excluding torch which is already installed)
RUN pip3 install \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "xformers==0.0.27.post2" \
    "ray>=2.9" \
    "nvidia-ml-py" \
    "sentencepiece" \
    "fastapi" \
    "uvicorn[standard]" \
    "pydantic>=2.8" \
    "openai>=1.0" \
    "prometheus_client>=0.18.0" \
    "pillow" \
    "numpy<2.0.0" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    "pyzmq" \
    "msgspec" \
    "psutil" \
    "py-cpuinfo" \
    "aiohttp" \
    "requests" \
    "tqdm"

# Install vllm-flash-attn for attention optimization
RUN pip3 install vllm-flash-attn==2.6.1

# Clone SGLang repository and checkout EXACT commit (hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 62f15eea5a0b4266cdae965d0337fd33f6673736

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="62f15eea5a0b4266cdae965d0337fd33f6673736" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Clean pip cache
RUN pip3 cache purge

# Verify installations
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Reset to interactive mode
ENV DEBIAN_FRONTEND=interactive

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
