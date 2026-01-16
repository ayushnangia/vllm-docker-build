# Dockerfile for SGLang
# Commit date: February 3, 2024
# Architecture: linux/amd64 (GPU)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st SHA occurrence: ENV declaration
ENV SGLANG_COMMIT=bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Verify Python version (base image should have 3.10)
RUN python --version

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with versions discovered from PyPI for 2024-02-03 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-02-03 era
# These were the actual versions available on Feb 3, 2024
fastapi==0.109.1
uvicorn==0.27.0.post1
pydantic==2.6.0
typing_extensions==4.9.0
outlines==0.0.25
pyzmq==25.1.2
aiohttp==3.9.3
transformers==4.37.2
numba==0.59.0
interegular==0.3.3
lark==1.1.9
rpyc==5.3.1
EOF

# Install PyTorch (base image should have it but ensure correct version)
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.3.0 (latest available on Feb 3, 2024) with --no-deps first
RUN pip install vllm==0.3.0 --no-deps

# Install vLLM dependencies from requirements.txt discovered from vLLM 0.3.0 repo
RUN pip install -c /opt/constraints.txt \
    ninja \
    psutil \
    "ray>=2.9" \
    sentencepiece \
    numpy \
    "transformers>=4.37.0" \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "aioprometheus[starlette]" \
    "pynvml==11.5.0"

# Install xformers FIRST with --no-deps to prevent pulling wrong torch
RUN pip install xformers==0.0.23.post1 --no-deps

# Install flashinfer from wheel index (not available on standard PyPI)
RUN pip install flashinfer --index-url https://flashinfer.ai/whl/cu121/torch2.1/

# Clone SGLang repository at specific commit
WORKDIR /sgl-workspace
# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# Verify commit SHA and write to file
# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="bb3a3b6675b1844a13ebe368ad693f3dc75b315b" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang package with --no-deps first
RUN cd /sgl-workspace/sglang && \
    pip install -e ./python --no-deps

# Install SGLang dependencies from pyproject.toml
# Note: sgl-kernel didn't exist yet (first released April 2025)
# NOTE: Do NOT reinstall torch - keep the 2.1.2 from base image
RUN pip install -c /opt/constraints.txt \
    requests \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    uvloop \
    uvicorn \
    pyzmq \
    interegular \
    lark \
    numba \
    pydantic \
    referencing \
    diskcache \
    cloudpickle \
    pillow

# Install optional dependencies for openai and anthropic
RUN pip install "openai>=1.0" numpy anthropic || true

# Install outlines (required by SGLang for structured generation)
RUN pip install -c /opt/constraints.txt outlines==0.0.25

# Final verification of all imports
RUN python -c "import torch; print(f'torch: {torch.__version__}')" && \
    python -c "import sglang; print('SGLang import OK')" && \
    python -c "import flashinfer; print('flashinfer OK')" && \
    python -c "import vllm; print('vLLM OK')" && \
    python -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python -c "import outlines; print('outlines OK')" && \
    python -c "import pydantic; print(f'pydantic: {pydantic.__version__}')"

WORKDIR /sgl-workspace
CMD ["python", "-c", "import sglang; print('SGLang loaded successfully')"]