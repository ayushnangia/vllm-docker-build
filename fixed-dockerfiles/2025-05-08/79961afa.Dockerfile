# Base image for torch 2.6.x with CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libibverbs-dev \
    software-properties-common \
    sudo \
    rdma-core \
    infiniband-diags \
    openssh-server \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip3 point to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip, setuptools, wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Pre-install torch 2.6.0 with CUDA 12.4 support BEFORE other packages
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python from the wheel repository (available for torch 2.6 + CUDA 12.4)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python

# Install sgl-kernel from PyPI (version 0.1.1 is available)
RUN pip3 install sgl-kernel==0.1.1

# Install critical dependencies from pyproject.toml
RUN pip3 install \
    transformers==4.51.1 \
    "torchao>=0.9.0" \
    datamodel_code_generator \
    ninja \
    packaging \
    numpy

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=79961afa8281f98f380d11db45c8d4b6e66a574f

# Clone SGLang repository and checkout the EXACT commit (occurrence 2 of 3)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 79961afa8281f98f380d11db45c8d4b6e66a574f

# VERIFY the checkout - compare against HARDCODED expected value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="79961afa8281f98f380d11db45c8d4b6e66a574f" && \
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
    sed -i 's/,,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[srt]"

# Install additional runtime dependencies
RUN pip3 install \
    aiohttp \
    requests \
    tqdm \
    IPython \
    setproctitle \
    compressed-tensors \
    datasets \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    "llguidance>=0.7.11,<0.8.0" \
    modelscope \
    orjson \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    "pyzmq>=25.1.2" \
    "soundfile==0.13.1" \
    uvicorn \
    uvloop \
    "xgrammar==0.1.17" \
    "blobfile==3.0.0" \
    cuda-python \
    "outlines>=0.0.44,<=0.1.11" \
    partial_json_parser \
    einops

# Replace system Triton with nightly (common pattern in SGLang Dockerfiles)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Set TORCH_CUDA_ARCH_LIST for H100 target
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Final verification of commit proof file
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists at /opt/sglang_commit.txt" && \
    cat /opt/sglang_commit.txt

ENV DEBIAN_FRONTEND=interactive

# Set default entrypoint
WORKDIR /sgl-workspace
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]