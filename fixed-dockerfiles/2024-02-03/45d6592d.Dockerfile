# Dockerfile for SGLang commit 45d6592d4053fe8b2b8dc9440f64c900de040d09
# Date: 2024-02-03
# SGLang version: 0.1.11 (requires vllm>=0.2.5)
# Architecture: linux/amd64 (GPU)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HARDCODED commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=45d6592d4053fe8b2b8dc9440f64c900de040d09

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with versions discovered from PyPI for Feb 2024 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-02-03 era
fastapi==0.109.0
uvicorn==0.26.0
pydantic==1.10.13
typing_extensions==4.9.0
pyzmq==25.1.2
aiohttp==3.9.3
psutil==5.9.8
rpyc==6.0.0
uvloop==0.19.0
interegular==0.3.3
lark==1.1.9
numba==0.59.0
referencing==0.33.0
diskcache==5.6.3
cloudpickle==3.0.0
pillow==10.2.0
numpy==1.26.3
sentencepiece==0.1.99
transformers==4.37.2
ray==2.9.1
EOF

# Pre-install torch (already in base image, but ensure correct version)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.1.2 (vLLM 0.2.7 requires xformers==0.0.23.post1)
RUN pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.2.7 WITHOUT dependencies to avoid conflicts
RUN pip install vllm==0.2.7 --no-deps

# Install vLLM dependencies from vLLM 0.2.7 requirements.txt (discovered from exploration)
# vLLM 0.2.7 explicitly requires pydantic==1.10.13 (v1)
RUN pip install -c /opt/constraints.txt \
    ninja \
    psutil \
    "ray>=2.5.1" \
    sentencepiece \
    numpy \
    "transformers>=4.36.0" \
    fastapi \
    "uvicorn[standard]" \
    "pydantic==1.10.13" \
    "aioprometheus[starlette]"

# Clone SGLang repository (HARDCODED SHA occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 45d6592d4053fe8b2b8dc9440f64c900de040d09

# Verify correct commit (HARDCODED SHA occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="45d6592d4053fe8b2b8dc9440f64c900de040d09" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Build flashinfer from source (no prebuilt wheels for Python 3.10 at this commit date)
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.0.3 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MAX_JOBS=96 pip install . --no-build-isolation && \
    rm -rf /tmp/flashinfer

# Install SGLang package with --no-deps first
WORKDIR /sgl-workspace/sglang
RUN pip install -e python --no-deps

# Install SGLang dependencies from pyproject.toml with constraints
RUN pip install -c /opt/constraints.txt \
    requests \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    torch \
    uvloop \
    uvicorn \
    pyzmq \
    interegular \
    lark \
    numba \
    pydantic==1.10.13 \
    referencing \
    diskcache \
    cloudpickle \
    pillow \
    "openai>=1.0" \
    anthropic \
    numpy

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')"

WORKDIR /sgl-workspace
CMD ["python3", "-c", "import sglang; print('SGLang loaded successfully')"]