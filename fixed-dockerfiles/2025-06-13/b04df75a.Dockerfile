# Base image for torch 2.7.x - needs CUDA 12.6
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3 python3-pip python3-dev \
        git curl wget \
        build-essential cmake ninja-build \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# 1st occurrence: ENV with full SHA
ENV SGLANG_COMMIT=b04df75acdda5b99999c02820e64b5b005c07159

# Install torch 2.7.1 with CUDA 12.6
RUN pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install numpy and packaging first (needed for builds)
RUN pip3 install numpy packaging ninja

# Build flashinfer from source (no wheels for torch 2.7)
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6.post1 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    cd / && \
    rm -rf /tmp/flashinfer

# Install sgl-kernel from PyPI (0.1.8.post1 is available)
RUN pip3 install sgl-kernel==0.1.8.post1

# 2nd occurrence: Clone and checkout exact commit
RUN git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git checkout b04df75acdda5b99999c02820e64b5b005c07159

# 3rd occurrence: Verify commit and write to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b04df75acdda5b99999c02820e64b5b005c07159" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: got $ACTUAL, expected $EXPECTED" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang dependencies from runtime_common
RUN pip3 install \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    "llguidance>=0.7.11,<0.8.0" \
    modelscope \
    msgspec \
    orjson \
    partial_json_parser \
    pillow \
    "prometheus-client>=0.20.0" \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    "pyzmq>=25.1.2" \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    transformers==4.52.3 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19 \
    cuda-python \
    "outlines>=0.0.44,<=0.1.11" \
    einops

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"cuda-python",*//g' python/pyproject.toml && \
    sed -i 's/"outlines[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"einops",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"xgrammar[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip3 install -e "python[all]"

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" && \
    python3 -c "from sgl_kernel import dispatch_bgmv; print('sgl-kernel OK')"

# Set default command
CMD ["/bin/bash"]