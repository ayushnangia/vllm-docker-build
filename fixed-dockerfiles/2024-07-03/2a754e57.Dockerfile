FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st SHA occurrence: ENV
ENV SGLANG_COMMIT=2a754e57b052e249ed4f8572cb6f0069ba6a495e

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /sgl-workspace

# Create constraints file with discovered versions from July 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-07-03 era
# These were found via WebFetch on PyPI release history pages
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
typing_extensions==4.11.0
outlines==0.0.44
pyzmq==26.0.3
aiohttp==3.9.5
numpy==1.26.4
transformers==4.42.3
tokenizers==0.19.1
huggingface_hub==0.23.4
sentencepiece==0.2.0
psutil==5.9.8
pillow==10.3.0
prometheus_client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
tiktoken==0.7.0
lm-format-enforcer==0.10.1
filelock==3.14.0
py-cpuinfo==9.0.0
requests==2.32.3
EOF

# Install vLLM 0.5.0 with --no-deps first
RUN pip install --upgrade pip && \
    pip install vllm==0.5.0 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    transformers \
    tokenizers \
    fastapi \
    aiohttp \
    openai \
    uvicorn \
    pydantic \
    pillow \
    prometheus_client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing_extensions \
    filelock

# Install CUDA-specific vLLM dependencies
# Note: Don't reinstall torch - keep the one from base image
RUN pip install \
    ray==2.9.0 \
    nvidia-ml-py \
    vllm-flash-attn==2.5.9

# Install xformers with --no-deps to prevent pulling wrong torch version
RUN pip install xformers==0.0.26.post1 --no-deps

# Install flashinfer from wheels
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 2a754e57b052e249ed4f8572cb6f0069ba6a495e

# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="2a754e57b052e249ed4f8572cb6f0069ba6a495e" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: SHA mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang dependencies first (the core deps from pyproject.toml)
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang srt dependencies with constraints
# Note: Don't reinstall torch - keep the one from base image
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    pydantic \
    rpyc \
    uvicorn \
    uvloop \
    pyzmq

# Install additional dependencies
RUN pip install datasets

# Verify installation
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
RUN python -c "import sglang; print('SGLang installed successfully')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python -c "import outlines; print('Outlines installed successfully')"

# Set working directory
WORKDIR /sgl-workspace

# Add metadata labels
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.revision="${SGLANG_COMMIT}"
LABEL org.opencontainers.image.description="SGLang Docker image for July 2024"

CMD ["/bin/bash"]