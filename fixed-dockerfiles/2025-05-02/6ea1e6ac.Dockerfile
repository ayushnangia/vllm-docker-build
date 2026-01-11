# Base image for torch 2.6.x with CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 on Ubuntu 22.04 using apt
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 with CUDA 12.4 (as specified in pyproject.toml)
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python==0.2.5 from flashinfer.ai wheels (available for cu124/torch2.6)
RUN pip3 install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel==0.1.1 from PyPI (verified available)
RUN pip3 install sgl-kernel==0.1.1

# Install transformers==4.51.1 (as specified in pyproject.toml)
RUN pip3 install transformers==4.51.1

# Install other required dependencies from pyproject.toml
RUN pip3 install packaging numpy ninja \
    "torchao>=0.9.0" \
    "xgrammar==0.1.17" \
    "cuda-python" \
    "outlines>=0.0.44,<=0.1.11" \
    "partial_json_parser" \
    "einops"

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    # Clean up any double commas or trailing commas
    sed -i 's/,,/,/g' python/pyproject.toml && \
    sed -i 's/,]/]/g' python/pyproject.toml

# Install SGLang from source in editable mode
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Set TORCH_CUDA_ARCH_LIST for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} OK')"

# Entry point
WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]