# Base image for torch 2.3.0 with CUDA 12.1 (vLLM 0.5.1 requires torch 2.3.0)
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget vim \
    build-essential ccache \
    libbz2-dev libffi-dev libgdbm-dev libnss3-dev \
    libncurses5-dev libreadline-dev libsqlite3-dev \
    libssl-dev zlib1g-dev software-properties-common \
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

# CUDA compatibility workaround
RUN ldconfig /usr/local/cuda-12.1/compat/

# HARDCODE the commit SHA (don't use ARG to avoid forgotten --build-arg issues)
ENV SGLANG_COMMIT=e1792cca2491af86f29782a3b83533a6566ac75b

# Pre-install torch 2.3.0 with CUDA 12.1 (vLLM 0.5.1 requires exactly torch 2.3.0)
RUN pip3 install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.1 with --no-deps to avoid dependency conflicts
RUN pip3 install vllm==0.5.1 --no-deps

# Install vLLM dependencies manually (based on vLLM 0.5.1 requirements)
RUN pip3 install \
    "transformers>=4.42.0" \
    "tokenizers>=0.19.1" \
    "sentencepiece" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.1" \
    "outlines>=0.0.43" \
    "pillow" \
    "prometheus-client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "requests" \
    "tqdm" \
    "filelock>=3.10.4" \
    "typing-extensions" \
    "py-cpuinfo" \
    "fastapi" \
    "aiohttp" \
    "openai" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "psutil" \
    "numpy<2.0.0" \
    "nvidia-ml-py" \
    "cmake>=3.21" \
    "ninja" \
    "ray>=2.9"

# Install xformers 0.0.26.post1 (exact version for torch 2.3.0)
RUN pip3 install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vllm-flash-attn (required by vLLM 0.5.1)
RUN pip3 install vllm-flash-attn==2.5.9

# Install flashinfer from wheels (available for torch 2.3 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# Clone SGLang repo and checkout EXACT commit (hardcoded, not variable)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout e1792cca2491af86f29782a3b83533a6566ac75b

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e1792cca2491af86f29782a3b83533a6566ac75b" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install additional dependencies that SGLang needs
RUN pip3 install \
    "hf_transfer" \
    "huggingface_hub" \
    "interegular" \
    "packaging" \
    "uvloop" \
    "zmq"

# Install SGLang from checked-out source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace triton with triton-nightly (common pattern in SGLang Dockerfiles)
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installations
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm import OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Verify CUDA availability
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK')"

# Set working directory
WORKDIR /sgl-workspace

ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]