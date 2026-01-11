# Base image for torch 2.7.1 - using CUDA 12.4 for recent torch versions
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=e3ec6bf4b65a50e26e936a96adc7acc618292002
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.7.1 (comes with CUDA 12.4 bundled on main PyPI)
RUN pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Install xformers compatible with torch 2.7
RUN pip3 install xformers==0.0.30

# Build flashinfer from source (no prebuilt wheel for torch 2.7)
RUN pip3 install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6.post1 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    cd / && \
    rm -rf /tmp/flashinfer

# Install sgl-kernel from PyPI (version is available)
RUN pip3 install sgl-kernel==0.1.8.post1

# Clone SGLang at the exact commit (2nd occurrence of SHA)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout e3ec6bf4b65a50e26e936a96adc7acc618292002

# Verify commit SHA (3rd occurrence of SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e3ec6bf4b65a50e26e936a96adc7acc618292002" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: Expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove pre-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip3 install -e "python[srt]"

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import xformers; print('xformers import OK')"

# Set entrypoint
WORKDIR /sgl-workspace
CMD ["/bin/bash"]