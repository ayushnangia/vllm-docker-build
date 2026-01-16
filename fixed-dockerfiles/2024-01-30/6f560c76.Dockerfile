# Fixed Dockerfile for sglang commit 6f560c761b2fc2f577682d0cfda62630f37a3bb0 (PR #117 - January 30, 2024)
# "Improve the control of streaming and improve the first token latency in streaming"
#
# Dependencies discovered via PyPI for January 30, 2024:
# - vllm 0.2.5 requires pydantic==1.10.13 (v1, NOT v2)
# - vllm 0.2.5 requires torch >=2.1.1 (we use 2.1.2 from Dec 14, 2023)
# - transformers 4.37.2 (released Jan 29, 2024)
# - fastapi 0.109.1 (released Jan 22, 2024)
# - uvicorn 0.27.0.post1 (released Jan 29, 2024)
# - typing_extensions 4.9.0 (released Dec 10, 2023)
# - pyzmq 25.1.2 (released Dec 5, 2023)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Build settings
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st hardcoded SHA: ENV variable
ENV SGLANG_COMMIT=6f560c761b2fc2f577682d0cfda62630f37a3bb0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Ensure correct torch version from base image
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with versions discovered from PyPI for Jan 30, 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-01-30 era
# vllm 0.2.5 requires pydantic v1, NOT v2
fastapi==0.109.1
uvicorn==0.27.0.post1
pydantic==1.10.13
typing_extensions==4.9.0
pyzmq==25.1.2
transformers==4.37.2
xformers==0.0.23.post1
EOF

# Install vllm 0.2.5 with --no-deps to avoid dependency conflicts
RUN pip install vllm==0.2.5 --no-deps

# Install xformers FIRST with --no-deps to prevent it from pulling torch 2.9.x
# xformers 0.0.23.post1 is compatible with torch 2.1.2
RUN pip install xformers==0.0.23.post1 --no-deps

# Install vllm's actual dependencies with constraints (excluding xformers which is already installed)
RUN pip install -c /opt/constraints.txt \
    ninja \
    psutil \
    "ray>=2.5.1,<2.10" \
    pandas \
    pyarrow \
    sentencepiece \
    "numpy<2.0" \
    "transformers==4.37.2" \
    fastapi==0.109.1 \
    "uvicorn[standard]==0.27.0.post1" \
    "pydantic==1.10.13" \
    "aioprometheus[starlette]"

# Set workspace
WORKDIR /sgl-workspace

# 2nd hardcoded SHA: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 6f560c761b2fc2f577682d0cfda62630f37a3bb0

# 3rd hardcoded SHA: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="6f560c761b2fc2f577682d0cfda62630f37a3bb0" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL ERROR: Expected commit $EXPECTED but got $ACTUAL"; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install sglang with --no-deps to control dependencies
WORKDIR /sgl-workspace/sglang
RUN pip install -e python --no-deps

# Install sglang's dependencies from pyproject.toml[srt] with constraints
# NOTE: Do NOT reinstall torch here - keep the 2.1.2 from base image
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
    pillow

# Install optional dependencies for full functionality
RUN pip install \
    "openai>=1.0" \
    anthropic \
    numpy

# Optional: Try to install flashinfer if wheels exist (non-critical if fails)
# flashinfer is optional at this early commit
RUN pip install flashinfer-cu121 || echo "flashinfer wheels not available, continuing without it"

# Verify installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

# Sanity check: Ensure pydantic is v1 (1.10.13) not v2
RUN python3 -c "import pydantic; assert pydantic.__version__.startswith('1.'), f'ERROR: pydantic {pydantic.__version__} is not v1'"

WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["/bin/bash"]