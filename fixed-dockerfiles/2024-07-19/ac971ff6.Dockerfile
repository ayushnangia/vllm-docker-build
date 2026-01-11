# Fixed Dockerfile for SGLang
# Date: 2024-07-19
# vLLM 0.5.1 requires torch 2.3.0

# Use pytorch base image for torch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    sudo \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install torch 2.3.0 (already in base image, but ensure correct version)
RUN pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.1 with --no-deps to avoid pulling wrong torch version
RUN pip3 install vllm==0.5.1 --no-deps

# Install vLLM dependencies manually (based on vLLM 0.5.1 requirements-cuda.txt)
RUN pip3 install \
    "ray>=2.9" \
    "nvidia-ml-py" \
    "xformers==0.0.26.post1" \
    "vllm-flash-attn==2.5.9" \
    "torchvision==0.18.0"

# Install common vLLM dependencies
RUN pip3 install \
    "cmake>=3.21" \
    "ninja" \
    "psutil" \
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
    "filelock>=3.10.4"

# Install flashinfer from wheels (available for torch 2.3.0 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# HARDCODE the commit SHA - first occurrence
ENV SGLANG_COMMIT=ac971ff633de330de3ded7f7475caaf7cd5bbdcd

# Clone SGLang and checkout EXACT commit (SHA is hardcoded, not variable) - second occurrence
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ac971ff633de330de3ded7f7475caaf7cd5bbdcd

# VERIFY the checkout - compare against HARDCODED expected value - third occurrence
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ac971ff633de330de3ded7f7475caaf7cd5bbdcd" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove flashinfer (already installed)
RUN sed -i 's/"flashinfer[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from checked-out source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace Triton with nightly version (as in original Dockerfile)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verification steps
RUN python3 -c "import torch; print(f'torch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Final verification of commit
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists: $(cat /opt/sglang_commit.txt)"

# Set working directory
WORKDIR /sgl-workspace

# Set entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]