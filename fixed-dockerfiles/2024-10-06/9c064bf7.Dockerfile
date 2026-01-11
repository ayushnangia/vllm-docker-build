# Fixed Dockerfile for SGLang commit 9c064bf78af8558dbc50fbd809f65dcafd6fd965 (2024-10-06)
# torch 2.4.0 requires pytorch base image for best compatibility
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# 1st SHA occurrence: ENV (hardcoded, not ARG)
ENV SGLANG_COMMIT=9c064bf78af8558dbc50fbd809f65dcafd6fd965

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /sgl-workspace

# Create constraints file with versions discovered from PyPI for October 2024 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-10-06 era via WebFetch
# Critical: outlines 0.0.44 requires pydantic>=2.0 (discovered via wheel inspection)
fastapi==0.115.0
uvicorn==0.31.0
pydantic==2.9.2
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
aiohttp==3.10.9
# Other dependencies from vLLM 0.5.5 requirements
numpy<2.0.0
transformers>=4.43.2
tokenizers>=0.19.1
openai>=1.0
pillow
prometheus_client>=0.18.0
prometheus-fastapi-instrumentator>=7.0.0
tiktoken>=0.6.0
lm-format-enforcer==0.10.6
filelock>=3.10.4
msgspec
librosa
soundfile
gguf==0.9.1
importlib_metadata
# CUDA-specific from vLLM 0.5.5
ray>=2.9
nvidia-ml-py
torchvision==0.19
xformers==0.0.27.post2
vllm-flash-attn==2.6.1
EOF

# Install vLLM 0.5.5 with --no-deps first
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    psutil sentencepiece numpy requests tqdm py-cpuinfo \
    transformers tokenizers protobuf fastapi aiohttp openai \
    uvicorn pydantic pillow prometheus_client prometheus-fastapi-instrumentator \
    tiktoken lm-format-enforcer typing_extensions filelock pyzmq msgspec \
    librosa soundfile gguf importlib_metadata \
    ray nvidia-ml-py torchvision xformers vllm-flash-attn

# Install outlines with constraints (must be 0.0.44, not newer)
RUN pip install -c /opt/constraints.txt outlines==0.0.44

# Clone SGLang at specific commit
# 2nd SHA occurrence: git checkout (hardcoded, not variable)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 9c064bf78af8558dbc50fbd809f65dcafd6fd965

# Verify commit SHA is correct
# 3rd SHA occurrence: verification (hardcoded expected value)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9c064bf78af8558dbc50fbd809f65dcafd6fd965" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "SUCCESS: Verified SGLang at commit $ACTUAL"

# Install SGLang with --no-deps first (editable install)
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang-specific dependencies from pyproject.toml
RUN pip install -c /opt/constraints.txt \
    aiohttp decord fastapi hf_transfer huggingface_hub interegular \
    packaging pillow psutil pydantic python-multipart \
    torch torchao uvicorn uvloop pyzmq modelscope

# Install additional outlines dependencies (discovered from wheel metadata)
RUN pip install -c /opt/constraints.txt \
    interegular jinja2 lark nest-asyncio cloudpickle diskcache \
    numba referencing jsonschema requests tqdm datasets pycountry pyairports

# Install flashinfer from wheel index (may fail if not available)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ || \
    echo "WARNING: flashinfer wheel not available, may need to build from source"

# Install sgl-kernel (check if version exists for this era)
RUN pip install sgl-kernel || \
    echo "WARNING: sgl-kernel not available, may need to build from source"

# Install other optional dependencies
RUN pip install datamodel_code_generator || true  # For openbmb/MiniCPM models

# Final verification - all imports should work
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import outlines; print('Outlines import OK')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import typing_extensions; print(f'typing_extensions version: {typing_extensions.__version__}')" && \
    echo "Commit proof:" && \
    cat /opt/sglang_commit.txt

# Cache cleanup
RUN pip cache purge

# Set environment for runtime
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace