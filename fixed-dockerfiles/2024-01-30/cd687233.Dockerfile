# Fixed Dockerfile for sglang commit cd6872334e9ead684049b8fccd5f2dac9433b1b4
# Date: 2024-01-30
# "Fix Mistral model loading (#108)"
#
# Key discovered versions from PyPI (via WebFetch):
# - vLLM 0.2.5 requires pydantic==1.10.13 (confirmed via repo exploration)
# - fastapi==0.109.0 (Jan 11, 2024 - supports both pydantic v1 and v2)
# - uvicorn==0.27.0.post1 (Jan 29, 2024)
# - pyzmq==25.1.2 (Dec 5, 2023)
# - typing_extensions==4.9.0 (Dec 10, 2023)
# - flashinfer wheels available for torch 2.1
# - sgl-kernel didn't exist yet (first released April 2025)
# - outlines requires pydantic v2, so we skip it to avoid conflicts

FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Build environment
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# SGLang commit SHA (1st occurrence)
ENV SGLANG_COMMIT=cd6872334e9ead684049b8fccd5f2dac9433b1b4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with discovered versions
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-01-30 era via WebFetch
fastapi==0.109.0
uvicorn[standard]==0.27.0.post1
pydantic==1.10.13
typing_extensions==4.9.0
pyzmq==25.1.2
# Pinned by vLLM and SGLang
transformers==4.36.2
# Dependencies
aiohttp
psutil
rpyc
uvloop
interegular
lark
numba
diskcache
cloudpickle
pillow
EOF

# Install vLLM 0.2.5 with --no-deps to avoid dependency conflicts
RUN pip install vllm==0.2.5 --no-deps

# Install xformers FIRST with --no-deps to prevent it from pulling wrong torch version
RUN pip install xformers==0.0.23 --no-deps

# Install vLLM dependencies from requirements.txt (discovered via repo exploration)
# NOTE: xformers already installed, torch comes from base image
RUN pip install -c /opt/constraints.txt \
    ninja \
    psutil \
    "ray>=2.5.1,<2.10" \
    pandas \
    pyarrow \
    sentencepiece \
    "numpy<2.0" \
    "transformers==4.36.2" \
    fastapi==0.109.0 \
    "uvicorn[standard]==0.27.0.post1" \
    "pydantic==1.10.13" \
    "aioprometheus[starlette]"

# Install flashinfer from wheels (available for torch 2.1)
RUN pip install flashinfer --index-url https://flashinfer.ai/whl/cu121/torch2.1/

# Set workspace
WORKDIR /sgl-workspace

# Clone SGLang at specific commit (2nd occurrence of SHA)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout cd6872334e9ead684049b8fccd5f2dac9433b1b4

# Verify commit SHA (3rd occurrence) and write proof
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="cd6872334e9ead684049b8fccd5f2dac9433b1b4" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "Error: Expected commit $EXPECTED but got $ACTUAL" >&2 && \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang from source with --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# Install SGLang dependencies (from pyproject.toml discovered via repo exploration)
# NOTE: Do NOT reinstall torch - keep the 2.1.1 from base image
RUN pip install -c /opt/constraints.txt \
    requests \
    aiohttp \
    psutil \
    rpyc \
    uvloop \
    "pyzmq==25.1.2" \
    interegular \
    lark \
    numba \
    diskcache \
    cloudpickle \
    pillow \
    openai

# Verify installations
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import flashinfer; print('FlashInfer imported successfully')"

# Sanity check: verify pip list shows correct packages
RUN pip list | grep -E "sglang|vllm|pydantic|fastapi|torch"

WORKDIR /sgl-workspace/sglang