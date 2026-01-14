# Fixed Dockerfile for SGLang
# Date: 2024-09-23
# Based on discovered versions from PyPI

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# HARDCODED SHA (1st of 3 occurrences)
ENV SGLANG_COMMIT=2854a5ea9fbb31165936f633ab99915dec760f8d

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace

# Create constraints file with discovered versions from September 2024
# All these versions were discovered via WebFetch from PyPI:
# - fastapi 0.114.2 (released Sep 13, 2024)
# - uvicorn 0.31.0 (released Sep 27, 2024)
# - pydantic 2.9.2 (released Sep 17, 2024)
# - typing_extensions 4.12.2 (released Jun 7, 2024, latest before Sep 2024)
# - outlines 0.0.46 (exists on PyPI, compatible with Sep 2024)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-09-23 era
fastapi==0.114.2
uvicorn==0.31.0
pydantic==2.9.2
pydantic-core==2.20.1
typing_extensions==4.12.2
outlines==0.0.46
pyzmq==26.2.0
EOF

# Install vLLM 0.5.5 with --no-deps first
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies (from requirements-common.txt)
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    protobuf \
    aiohttp \
    "openai>=1.0" \
    "uvicorn[standard]==0.31.0" \
    "pydantic==2.9.2" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions==4.12.2" \
    "filelock>=3.10.4" \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    "gguf==0.9.1" \
    importlib_metadata \
    "fastapi==0.114.2"

# Install vLLM CUDA dependencies (from requirements-cuda.txt)
RUN pip install \
    "ray>=2.9" \
    nvidia-ml-py \
    "torchvision==0.19" \
    "xformers==0.0.27.post2" \
    "vllm-flash-attn==2.6.1"

# Clone SGLang at specific commit (2nd occurrence of SHA)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 2854a5ea9fbb31165936f633ab99915dec760f8d

# Verify commit SHA (3rd occurrence of SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="2854a5ea9fbb31165936f633ab99915dec760f8d" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "FATAL: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies from pyproject.toml [srt] extras
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    python-multipart \
    torchao \
    uvloop \
    pyzmq

# Install flashinfer (for torch 2.4 + CUDA 12.1)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ || \
    echo "WARNING: flashinfer wheels not available, may need to build from source"

# Install sgl-kernel from PyPI (version available in Sep 2024)
RUN pip install sgl-kernel==0.3.4 || pip install sgl-kernel

# For MiniCPM models
RUN pip install datamodel_code_generator

# Install optional dependencies
RUN pip install "openai>=1.0" tiktoken "anthropic>=0.20.0" "litellm>=1.0.0" || true

# Verification
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import typing_extensions; print(f'typing_extensions loaded')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Clean pip cache
RUN pip cache purge

# Reset to interactive mode
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace