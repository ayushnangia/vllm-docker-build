# Base image for torch 2.6.x with CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    software-properties-common \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.6.0 (pinning to avoid 2.7.1 conflicts)
RUN pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install sgl-kernel from PyPI (version 0.1.7 is available)
RUN pip3 install sgl-kernel==0.1.7

# Build flashinfer from source (no wheel for 0.2.6.post1 with torch 2.6)
RUN pip3 install ninja numpy packaging \
    && git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout v0.2.6 \
    && cd python \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . \
    && rm -rf /tmp/flashinfer

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=22a6b9fc051154347b6eb5064d2f6ef9b4dba471

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 22a6b9fc051154347b6eb5064d2f6ef9b4dba471

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="22a6b9fc051154347b6eb5064d2f6ef9b4dba471" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang/python && \
    sed -i 's/"flashinfer[^"]*",*//g' pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchaudio[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' pyproject.toml && \
    sed -i 's/"torchao==0.9.0"/"torchao>=0.12.0"/g' pyproject.toml

# Install SGLang from source
WORKDIR /sgl-workspace/sglang
RUN cd python && pip3 install -e ".[srt]"

# Install additional runtime dependencies
RUN pip3 install \
    transformers==4.52.3 \
    xgrammar==0.1.19 \
    orjson \
    msgspec \
    pydantic \
    fastapi \
    uvicorn \
    uvloop \
    prometheus-client \
    pyzmq \
    pillow \
    soundfile \
    scipy \
    einops \
    outlines

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import sgl_kernel; print('SGL kernel import OK')"

ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace/sglang

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]