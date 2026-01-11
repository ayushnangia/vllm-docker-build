# SGLang Docker image for commit 1acca3a2c685221cdb181c2abda4f635e1ead435
# Date: 2025-05-02
# SGLang version: 0.4.6.post2
# torch: 2.6.0, flashinfer_python: 0.2.5, sgl-kernel: 0.1.1

# Base image for torch 2.6.x - using CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# HARDCODE commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=1acca3a2c685221cdb181c2abda4f635e1ead435

# Install torch 2.6.0 with CUDA 12.4
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python 0.2.5 from flashinfer.ai wheels
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel 0.1.1 from PyPI
RUN pip3 install sgl-kernel==0.1.1

# Install other core dependencies from pyproject.toml
RUN pip3 install \
    numpy \
    packaging \
    ninja \
    transformers==4.51.1 \
    xgrammar==0.1.17 \
    "torchao>=0.9.0" \
    blobfile==3.0.0 \
    einops \
    partial_json_parser \
    "outlines>=0.0.44,<=0.1.11" \
    cuda-python

# Clone SGLang repo and checkout exact commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 1acca3a2c685221cdb181c2abda4f635e1ead435

# Verify the checkout matches expected commit (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="1acca3a2c685221cdb181c2abda4f635e1ead435" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"xgrammar[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"blobfile[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"cuda-python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"einops[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"partial_json_parser[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"outlines[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install triton-nightly (often needed for modern SGLang)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# For openbmb/MiniCPM models (from original Dockerfile)
RUN pip3 install datamodel_code_generator

# Set TORCH_CUDA_ARCH_LIST for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Set working directory
WORKDIR /sgl-workspace

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]