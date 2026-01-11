# SGLang Dockerfile for commit 22a6b9fc051154347b6eb5064d2f6ef9b4dba471 (2025-06-13)
# Build with: docker build -f 22a6b9fc.Dockerfile -t sglang:22a6b9fc .

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# HARDCODE 1/3: Set commit SHA in ENV
ENV SGLANG_COMMIT=22a6b9fc051154347b6eb5064d2f6ef9b4dba471
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip \
    git curl wget \
    build-essential \
    ninja-build \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install basic Python packages
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Create constraints file with discovered versions from PyPI
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-06-13 era
# Critical: outlines 0.0.44 requires pydantic>=2.0
# Critical: typing_extensions 4.13.2 (before 4.14.0 which added Sentinel)
fastapi==0.111.0
uvicorn==0.30.0
pydantic==2.11.6
typing_extensions==4.13.2
outlines==0.0.44
pyzmq==26.2.0
EOF

WORKDIR /sgl-workspace

# Install PyTorch 2.7.1 with CUDA 12.4
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# Build settings for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Clone and install flashinfer from source (no wheels for torch 2.7)
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    pip install ninja numpy pybind11 && \
    pip install -e . --no-deps

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.7

# HARDCODE 2/3: Clone SGLang at specific commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 22a6b9fc051154347b6eb5064d2f6ef9b4dba471

# HARDCODE 3/3: Verify commit SHA and save to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="22a6b9fc051154347b6eb5064d2f6ef9b4dba471" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang core without dependencies
RUN cd /sgl-workspace/sglang && \
    pip install -e python --no-deps

# Install runtime_common dependencies from pyproject.toml with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.11 \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client==0.20.0 \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq==26.2.0 \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    transformers==4.52.3 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19 \
    cuda-python \
    outlines==0.0.44 \
    einops

# Verify installation
RUN python3.10 -c "import sglang; print('SGLang imported successfully')" && \
    python3.10 -c "import flashinfer; print('Flashinfer imported successfully')" && \
    python3.10 -c "import outlines; print('Outlines imported successfully')" && \
    python3.10 -c "import pydantic; print(f'Pydantic version: {pydantic.VERSION}')" && \
    cat /opt/sglang_commit.txt

# Set the default command
CMD ["python3.10", "-m", "sglang.launch_server"]

ENV DEBIAN_FRONTEND=interactive