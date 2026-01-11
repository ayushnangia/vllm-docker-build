# Build from PyTorch 2.4.0 with CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=62f15eea5a0b4266cdae965d0337fd33f6673736
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    wget \
    sudo \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with discovered versions from PyPI (2024-09-05 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions discovered from PyPI for 2024-09-05
fastapi==0.112.0
uvicorn==0.30.6
pydantic==2.9.0
pydantic-core==2.23.0
typing_extensions==4.12.2
outlines==0.0.46
pyzmq==26.2.0
numpy<2.0.0
aiohttp==3.10.5
pillow==10.4.0
prometheus_client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
EOF

# Install vLLM 0.5.5 with --no-deps first
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies from requirements-common.txt
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    numpy \
    requests \
    tqdm \
    py-cpuinfo \
    transformers \
    tokenizers \
    protobuf \
    fastapi \
    aiohttp \
    openai \
    uvicorn \
    pydantic \
    pillow \
    prometheus_client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer==0.10.6 \
    typing_extensions \
    filelock \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    gguf==0.9.1 \
    importlib_metadata

# Install vLLM CUDA dependencies
RUN pip install \
    ray>=2.9 \
    nvidia-ml-py \
    torchvision==0.19.0 \
    xformers==0.0.27.post2 \
    vllm-flash-attn==2.6.1

# Install flashinfer from their wheel index
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Create workspace
WORKDIR /sgl-workspace

# Clone SGLang at the exact commit (1st commit reference)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 62f15eea5a0b4266cdae965d0337fd33f6673736

# Verify commit (2nd commit reference)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="62f15eea5a0b4266cdae965d0337fd33f6673736" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies from pyproject.toml[srt]
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    uvicorn \
    uvloop \
    pyzmq

# Additional optional dependencies
RUN pip install \
    openai>=1.0 \
    tiktoken \
    anthropic>=0.20.0 \
    litellm>=1.0.0

# Sanity check imports
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Final verification (uses ENV variable)
RUN COMMIT_FILE="/opt/sglang_commit.txt" && \
    EXPECTED="${SGLANG_COMMIT}" && \
    if [ -f "$COMMIT_FILE" ]; then \
        ACTUAL=$(cat "$COMMIT_FILE") && \
        if [ "$ACTUAL" = "$EXPECTED" ]; then \
            echo "✓ Commit verification passed: $ACTUAL" ; \
        else \
            echo "✗ Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
        fi \
    else \
        echo "✗ Commit file not found!" && exit 1; \
    fi

# Set the working directory to SGLang
WORKDIR /sgl-workspace/sglang

# Default command
CMD ["/bin/bash"]