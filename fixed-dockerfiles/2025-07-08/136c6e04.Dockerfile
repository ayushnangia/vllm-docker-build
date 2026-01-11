# Base image for torch 2.6.x with CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0"

# HARDCODE the commit SHA - this Dockerfile is specific to this commit
ENV SGLANG_COMMIT=136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 (latest available, since 2.7.1 doesn't exist)
# Using CUDA 12.4 index
RUN pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install numpy and packaging first (needed for flashinfer build)
RUN pip3 install numpy packaging

# Build flashinfer from source since 0.2.7.post1 wheel doesn't exist for torch 2.6
# Using v0.2.6 which is compatible with torch 2.6
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip3 install --no-build-isolation . && \
    cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel 0.2.4 from PyPI (it exists)
RUN pip3 install sgl-kernel==0.2.4

# Clone SGLang repository and checkout the EXACT commit (hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7

# VERIFY the checkout - compare against HARDCODED expected value
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="136c6e0431c2067c3a2a98ad2c77fc89a9cb98e7" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
# Remove flashinfer_python, sgl-kernel, and torch versions since we pre-installed them
RUN cd /sgl-workspace/sglang/python && \
    cp pyproject.toml pyproject.toml.orig && \
    sed -i 's/"flashinfer_python[^"]*",*//g' pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' pyproject.toml && \
    sed -i '/^[[:space:]]*,$/d' pyproject.toml && \
    sed -i 's/,\]/]/g' pyproject.toml

# Install torchao (specified in pyproject.toml, compatible with torch 2.6)
RUN pip3 install torchao==0.9.0

# Install SGLang from the checked-out source
# Note: pyproject.toml is in python/ subdirectory for this commit
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install additional runtime dependencies that may be needed
RUN pip3 install \
    transformers==4.53.0 \
    xgrammar==0.1.19 \
    timm==1.0.16 \
    outlines==0.1.11 \
    llguidance \
    cuda-python \
    einops

# Verify installations
RUN python3 -c "import torch; print(f'torch version: {torch.__version__}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl_kernel import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import transformers; print(f'transformers version: {transformers.__version__}')"

# Verify the commit SHA is correct by checking the file exists
RUN test -f /opt/sglang_commit.txt && \
    COMMIT=$(cat /opt/sglang_commit.txt) && \
    echo "Commit in container: $COMMIT"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]