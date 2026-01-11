# SGLang Dockerfile for commit 6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe
# Date: 2025-05-02
# torch 2.6.0 requires CUDA 12.4+ base image

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Build settings for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV DEBIAN_FRONTEND=noninteractive

# 1st hardcoded SHA occurrence: ENV
ENV SGLANG_COMMIT=6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install torch 2.6.0 with CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with versions discovered from PyPI for May 2025 era
# Versions discovered via WebFetch from PyPI:
# - fastapi 0.111.0 (released May 3, 2024)
# - uvicorn 0.29.0 (released March 20, 2024)
# - pydantic 2.7.1 (released April 23, 2024) - required by outlines>=2.0
# - typing_extensions 4.11.0 (released April 5, 2024) - before Sentinel addition
# - outlines 0.0.44 (released June 14, 2024) - minimum required by SGLang
# - pyzmq 26.0.3 (released May 1, 2024)
RUN cat > /opt/constraints.txt <<'EOF'
# Core packages - versions discovered from PyPI
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
typing_extensions==4.11.0
outlines==0.0.44
pyzmq==26.0.3

# Additional packages from requirements
transformers==4.51.1
tokenizers>=0.19.1
sentencepiece
numpy
requests
py-cpuinfo
psutil
filelock>=3.10.4
prometheus_client>=0.18.0
prometheus-fastapi-instrumentator>=7.0.0

# SGLang common dependencies
aiohttp
tqdm
IPython
setproctitle
compressed-tensors
datasets
decord
hf_transfer
huggingface_hub
interegular
llguidance>=0.7.11,<0.8.0
modelscope
ninja
orjson
packaging
pillow
python-multipart
soundfile==0.13.1
torchao>=0.9.0
uvloop
xgrammar==0.1.17
blobfile==3.0.0
einops
partial_json_parser
cuda-python
EOF

# Install core dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.1 \
    typing_extensions==4.11.0 \
    outlines==0.0.44 \
    pyzmq==26.0.3

# Install SGLang dependencies
RUN pip install -c /opt/constraints.txt \
    transformers==4.51.1 tokenizers sentencepiece numpy requests py-cpuinfo \
    psutil filelock prometheus_client prometheus-fastapi-instrumentator \
    aiohttp tqdm IPython setproctitle compressed-tensors datasets decord \
    hf_transfer huggingface_hub interegular llguidance modelscope ninja \
    orjson packaging pillow python-multipart soundfile torchao uvloop \
    xgrammar==0.1.17 blobfile==3.0.0 einops partial_json_parser cuda-python

# Install sgl-kernel 0.1.1 (found via WebFetch at https://pypi.org/simple/sgl-kernel/)
RUN pip install sgl-kernel==0.1.1

# Build flashinfer from source (no prebuilt wheels for torch 2.6 - verified via WebFetch)
WORKDIR /tmp
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.5 && \
    cd python && \
    pip install . && \
    cd / && rm -rf /tmp/flashinfer

# Clone and install SGLang
WORKDIR /sgl-workspace

# 2nd hardcoded SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe

# 3rd hardcoded SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang in editable mode without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Clean up
RUN rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

WORKDIR /sgl-workspace

# Set entry point
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]