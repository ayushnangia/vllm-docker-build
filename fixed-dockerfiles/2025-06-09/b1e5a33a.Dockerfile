# Fixed Dockerfile for SGLang commit b1e5a33ae337d20e35e966b8d82a02a913d32689
# Date: 2025-06-09
# Target: H100 benchmarking

# Base image for torch 2.6.0 with CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 with CUDA 12.4 (MUST be before other packages)
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python from wheel (available for cu124/torch2.6)
RUN pip3 install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel from PyPI (version 0.1.6.post1 is available)
RUN pip3 install sgl-kernel==0.1.6.post1

# Install other key dependencies from pyproject.toml
RUN pip3 install transformers==4.52.3 torchao==0.9.0

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=b1e5a33ae337d20e35e966b8d82a02a913d32689

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout b1e5a33ae337d20e35e966b8d82a02a913d32689

# VERIFY commit matches expected (3rd occurrence - hardcoded)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b1e5a33ae337d20e35e966b8d82a02a913d32689" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,,*/,/g; s/\[,/[/g; s/,\]/]/g' python/pyproject.toml

# Install SGLang from source (editable install)
WORKDIR /sgl-workspace/sglang
RUN cd python && pip3 install -e ".[srt]"

# Install Triton nightly (often needed for modern builds)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')"
RUN python3 -c "import flashinfer; print('Flashinfer import OK')"
RUN python3 -c "import sgl_kernel; print('sgl_kernel import OK')"
RUN python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.version.cuda}')"
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Final verification that commit proof file exists
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists: $(cat /opt/sglang_commit.txt)"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]