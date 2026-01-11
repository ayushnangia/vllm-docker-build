# SGLang Dockerfile for commit a191a0e47c2f0b0c8aed28080b9cb78624365e92
# Date: 2025-05-26 (torch 2.6.0+ requires ubuntu22.04)
# Key versions: torch 2.6.0, flashinfer 0.2.5, sgl-kernel 0.1.4, outlines 0.0.44 (requires pydantic v2)

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 1st hardcoded commit SHA (1 of 3)
ENV SGLANG_COMMIT=a191a0e47c2f0b0c8aed28080b9cb78624365e92

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ccache \
    cmake \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Update pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /sgl-workspace

# Set build environment for H100 support
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV NVCC_THREADS=8
ENV USE_FLASH_ATTN=1

# Install torch 2.6.0 with CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with versions discovered from PyPI for May 2025 era
# CRITICAL: outlines 0.0.44 requires pydantic >=2.0
# typing_extensions must be <4.14 to avoid Sentinel issues with older pydantic-core
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for May 2025 era
# Based on WebFetch research of release dates
fastapi==0.111.0
uvicorn==0.30.0
pydantic==2.7.1
pydantic-core==2.18.2
typing_extensions==4.11.0
outlines==0.0.44
pyzmq==26.0.3
aiohttp==3.9.5
numpy==1.26.4
tqdm==4.66.4
requests==2.31.0
setuptools==69.5.1
msgspec==0.18.6
orjson==3.10.3
ninja==1.11.1.1
packaging==24.0
pillow==10.3.0
psutil==5.9.8
scipy==1.13.1
einops==0.8.0
huggingface_hub==0.23.0
datasets==2.19.1
uvloop==0.19.0
blobfile==3.0.0
prometheus-client==0.20.0
pynvml==11.5.0
python-multipart==0.0.9
partial_json_parser==0.2.1.1
interegular==0.3.3
compressed-tensors==0.6.0
modelscope==1.14.0
hf_transfer==0.1.6
soundfile==0.13.1
EOF

# Install sgl-kernel 0.1.4 from PyPI (confirmed available via WebFetch)
RUN pip install sgl-kernel==0.1.4

# Install flashinfer from the specific wheel URL for cu124/torch2.6
# WebFetch confirmed 0.2.5 is available at this location
RUN pip install flashinfer_python==0.2.5 --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install torchao
RUN pip install torchao==0.9.0

# Clone SGLang at the specific commit (2nd hardcoded commit SHA - 2 of 3)
RUN git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git checkout a191a0e47c2f0b0c8aed28080b9cb78624365e92

# Verify the commit (3rd hardcoded commit SHA - 3 of 3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="a191a0e47c2f0b0c8aed28080b9cb78624365e92" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install base SGLang package first with --no-deps to avoid version conflicts
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install runtime_common dependencies with constraints
# Using versions discovered via WebFetch from PyPI
RUN pip install -c /opt/constraints.txt \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    soundfile==0.13.1 \
    scipy \
    uvicorn \
    uvloop

# Install transformers and xgrammar with specific versions from pyproject.toml
RUN pip install transformers==4.51.1 xgrammar==0.1.19

# Install llguidance
RUN pip install "llguidance>=0.7.11,<0.8.0"

# Install other core dependencies
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    cuda-python \
    einops

# Install outlines with constraints (MUST use pydantic v2)
RUN pip install -c /opt/constraints.txt outlines==0.0.44

# Sanity checks - verify all critical imports work
RUN python3 -c "import sglang; print(f'✓ SGLang version: {sglang.__version__}')"
RUN python3 -c "import flashinfer; print('✓ FlashInfer imported successfully')"
RUN python3 -c "import sgl_kernel; print('✓ sgl_kernel imported successfully')"
RUN python3 -c "import torch; print(f'✓ Torch {torch.__version__} with CUDA {torch.version.cuda}')"
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Critical: Verify outlines works with pydantic v2
RUN python3 -c "import outlines; import pydantic; assert pydantic.VERSION.startswith('2.'), f'Wrong pydantic version: {pydantic.VERSION}'; print(f'✓ Outlines {outlines.__version__} with Pydantic {pydantic.VERSION}')"

# Verify typing_extensions doesn't have Sentinel (should be pre-4.14)
RUN python3 -c "import typing_extensions; hasattr(typing_extensions, 'Sentinel') and exit(1) or print('✓ typing_extensions OK (no Sentinel)')"

# Final verification that commit file exists and is correct
RUN test -f /opt/sglang_commit.txt && \
    [ "$(cat /opt/sglang_commit.txt)" = "a191a0e47c2f0b0c8aed28080b9cb78624365e92" ] && \
    echo "✓ Commit verification successful"

ENV DEBIAN_FRONTEND=interactive

# Set the working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]