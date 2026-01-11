# SGLang Dockerfile for commit 2a754e57b052e249ed4f8572cb6f0069ba6a495e
# Date: 2024-07-03
# SGLang version: 0.1.17
# vLLM: 0.5.0 (requires torch 2.3.0)

# Use pytorch base image with torch 2.3.0 pre-installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Ensure torch 2.3.0 is installed (required by vLLM 0.5.0)
# The base image already has torch 2.3.0, but let's ensure correct version
RUN pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install numpy and packaging first (common dependencies)
RUN pip3 install numpy packaging

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=2a754e57b052e249ed4f8572cb6f0069ba6a495e

# Clone SGLang repo and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 2a754e57b052e249ed4f8572cb6f0069ba6a495e

# VERIFY the commit matches (3rd occurrence - hardcoded)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="2a754e57b052e249ed4f8572cb6f0069ba6a495e" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install xformers 0.0.26.post1 for torch 2.3.x compatibility
RUN pip3 install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.0 without dependencies to avoid torch version conflicts
RUN pip3 install vllm==0.5.0 --no-deps

# Manually install vLLM dependencies with correct versions
RUN pip3 install \
    "cmake>=3.21" \
    ninja \
    psutil \
    sentencepiece \
    requests \
    py-cpuinfo \
    "transformers>=4.40.0" \
    "tokenizers>=0.19.1" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.1" \
    "outlines>=0.0.43" \
    fastapi \
    aiohttp \
    openai \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    typing_extensions \
    "filelock>=3.10.4" \
    nvidia-ml-py \
    ray

# Install additional SGLang dependencies from pyproject.toml
RUN pip3 install \
    hf_transfer \
    huggingface_hub \
    interegular \
    rpyc \
    uvloop \
    zmq \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Install SGLang from checked-out source
# Note: pyproject.toml is in python/ subdirectory for this commit
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Install additional packages for benchmark/serving
RUN pip3 install datasets

# Replace triton with triton-nightly for better compatibility
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verification step - ensure SGLang and dependencies import correctly
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')" && \
    python3 -c "import torch; assert torch.__version__.startswith('2.3.'), f'Wrong torch version: {torch.__version__}'" && \
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]