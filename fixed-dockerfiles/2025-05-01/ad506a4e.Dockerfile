# Fixed Dockerfile for SGLang commit from 2025-05-02
# Base image for torch 2.6.x
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# 1st SHA occurrence: ENV variable
ENV SGLANG_COMMIT=ad506a4e6bf3d9ac12100d4648c48df76f584c4e

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip and ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.6.0
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from May 2025
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for May 2025 era
fastapi==0.114.2
uvicorn==0.34.2
pydantic==2.11.5
typing_extensions==4.13.2
outlines==0.0.44
pyzmq==26.4.0
prometheus-client==0.22.0
transformers==4.51.1
huggingface-hub==0.23.0
aiohttp==3.9.5
requests==2.31.0
tqdm==4.68.0
numpy==1.26.4
IPython==8.25.0
setproctitle==1.3.3
EOF

# Install runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    compressed-tensors \
    datasets \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.11 \
    modelscope \
    ninja \
    orjson \
    packaging \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    soundfile==0.13.1 \
    torchao==0.9.0 \
    transformers \
    uvicorn \
    uvloop \
    xgrammar==0.1.17 \
    blobfile==3.0.0 \
    partial_json_parser \
    einops \
    cuda-python \
    outlines

# Install sgl-kernel 0.1.1 from PyPI
RUN pip install sgl-kernel==0.1.1

# Build flashinfer from source (no torch 2.6 wheels available on flashinfer.ai)
WORKDIR /tmp
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.5 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install . && \
    cd / && \
    rm -rf /tmp/flashinfer

# Clone SGLang at specific commit
WORKDIR /sgl-workspace
# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout ad506a4e6bf3d9ac12100d4648c48df76f584c4e

# Verify commit SHA and write to file
# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ad506a4e6bf3d9ac12100d4648c48df76f584c4e" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install datamodel_code_generator for MiniCPM models
RUN pip install datamodel_code_generator

# Sanity check - verify all imports work
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')" && \
    python3 -c "import torch; print(f'Torch {torch.__version__} with CUDA {torch.cuda.is_available()}')"

# Set working directory
WORKDIR /sgl-workspace

# Reset to interactive
ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]