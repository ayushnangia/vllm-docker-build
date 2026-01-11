# Fixed Dockerfile for SGLang commit 564a898ad975192b593be81387d11faf15cb1d3e
# Date: 2024-07-14
# vLLM: 0.5.1 (from pyproject.toml) - requires torch 2.3.0
# torch: 2.3.0 (required by vLLM 0.5.1)
# CUDA: 12.1

# Use pytorch base image with torch 2.3.0 pre-installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Ensure torch 2.3.0 is installed (should be in base image, but verify)
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install torchvision compatible with torch 2.3.0
RUN pip install torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.3.0
RUN pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.1 with --no-deps to avoid torch version conflicts
RUN pip install vllm==0.5.1 --no-deps

# Install vLLM dependencies manually (based on vLLM 0.5.1 requirements-cuda.txt)
RUN pip install \
    "nvidia-ml-py" \
    "ray>=2.9" \
    "sentencepiece" \
    "numpy<2.0.0" \
    "requests" \
    "tqdm" \
    "py-cpuinfo" \
    "transformers>=4.42.0" \
    "tokenizers>=0.19.1" \
    "fastapi" \
    "aiohttp" \
    "openai" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.1" \
    "outlines>=0.0.43" \
    "typing_extensions" \
    "filelock>=3.10.4" \
    "vllm-flash-attn==2.5.9" \
    "psutil" \
    "cmake>=3.21" \
    "packaging" \
    "huggingface_hub" \
    "hf_transfer" \
    "interegular" \
    "rpyc" \
    "uvloop" \
    "pyzmq"

# Install flashinfer from wheel for torch 2.3.0 + CUDA 12.1
RUN pip install flashinfer==0.2.0.post1 -i https://flashinfer.ai/whl/cu121/torch2.3/

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=564a898ad975192b593be81387d11faf15cb1d3e

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 564a898ad975192b593be81387d11faf15cb1d3e

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="564a898ad975192b593be81387d11faf15cb1d3e" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove flashinfer (since we installed it manually)
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Install Triton nightly (as in original Dockerfile)
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installations
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vllm OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]