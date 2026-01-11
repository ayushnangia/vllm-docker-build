# Fixed Dockerfile for SGLang commit 6fc17596 (2025-05-01)
# torch==2.6.0, flashinfer_python==0.2.5, sgl-kernel==0.1.1

# Base image for torch 2.6.x
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (Ubuntu 22.04 has it in default repos)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 with CUDA 12.4 index FIRST
RUN pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install torchvision (specified in pyproject.toml)
RUN pip3 install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer_python 0.2.5 from flashinfer.ai wheel index (EXACT version)
RUN pip3 install flashinfer_python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel 0.1.1 from PyPI
RUN pip3 install sgl-kernel==0.1.1

# Install runtime_common dependencies from pyproject.toml
RUN pip3 install \
    compressed-tensors \
    datasets \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    "llguidance>=0.7.11,<0.8.0" \
    modelscope \
    ninja \
    orjson \
    packaging \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    "pyzmq>=25.1.2" \
    "soundfile==0.13.1" \
    "torchao>=0.9.0" \
    "transformers==4.51.1" \
    uvicorn \
    uvloop \
    "xgrammar==0.1.17" \
    "blobfile==3.0.0"

# Install additional srt dependencies
RUN pip3 install \
    cuda-python \
    "outlines>=0.0.44,<=0.1.11" \
    partial_json_parser \
    einops

# Install for MiniCPM models
RUN pip3 install datamodel_code_generator

# HARDCODE the commit SHA (occurrence 1 of 3)
ENV SGLANG_COMMIT=6fc175968c3a9fc0521948aa3636887cd6d84107

# Clone SGLang and checkout EXACT commit (occurrence 2 of 3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 6fc175968c3a9fc0521948aa3636887cd6d84107

# VERIFY commit - compare against HARDCODED value (occurrence 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6fc175968c3a9fc0521948aa3636887cd6d84107" && \
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
    sed -i 's/"cuda-python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"outlines[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"partial_json_parser[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"einops[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"xgrammar[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"blobfile[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python"

# Install Triton nightly (common pattern in SGLang Dockerfiles)
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import OK')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Set environment for H100 architecture
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Reset frontend
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]