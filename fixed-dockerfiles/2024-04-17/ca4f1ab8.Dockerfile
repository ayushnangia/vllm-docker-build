# SGLang Docker Image for commit ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb (2024-04-17)
# PyTorch 2.1.2 with CUDA 12.1
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for build
ENV MAX_JOBS=96
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set work directory
WORKDIR /sgl-workspace

# Set the commit SHA as environment variable (1st occurrence - HARDCODED)
ENV SGLANG_COMMIT=ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb

# Create constraints file with versions discovered from PyPI for April 17, 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions from PyPI for April 2024
fastapi==0.110.2
uvicorn==0.29.0
pydantic==2.7.0
pydantic-core==2.18.1
typing_extensions==4.11.0
outlines==0.0.39
pyzmq==26.0.0
aiohttp==3.9.5

# From vLLM 0.3.3 requirements
torch==2.1.2
transformers==4.38.2
xformers==0.0.23.post1
ray==2.9.3
sentencepiece==0.2.0
numpy==1.26.4
psutil==5.9.8
prometheus_client==0.20.0
pynvml==11.5.0
triton==2.1.0
cupy-cuda12x==12.1.0

# Additional dependencies
uvloop==0.19.0
rpyc==6.0.0
interegular==0.3.3
pillow==10.3.0
requests==2.31.0
tqdm==4.66.2
EOF

# Install torch first (ensure correct version)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.3.3 without dependencies
RUN pip install vllm==0.3.3 --no-deps

# Install vLLM dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    ninja \
    psutil \
    ray \
    sentencepiece \
    numpy \
    transformers \
    xformers \
    fastapi \
    uvicorn \
    pydantic \
    pydantic-core \
    prometheus_client \
    pynvml \
    triton \
    outlines \
    cupy-cuda12x \
    typing_extensions

# Clone SGLang at the exact commit (2nd occurrence - HARDCODED)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb

# Verify commit SHA and save to file (3rd occurrence - HARDCODED)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "COMMIT MISMATCH: Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove vllm from dependencies since we already installed it
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang in editable mode without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    uvloop \
    uvicorn \
    pyzmq \
    interegular \
    pydantic \
    pillow \
    outlines \
    requests \
    tqdm \
    openai \
    anthropic \
    tiktoken

# Install datasets for benchmarking
RUN pip install datasets

# Verification step - ensure everything imports correctly
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import xformers; print('xformers import OK')" && \
    python3 -c "import outlines; print('outlines import OK')" && \
    python3 -c "import pydantic; print(f'pydantic: {pydantic.__version__}')"

# Final commit verification
RUN test -f /opt/sglang_commit.txt && \
    echo "SGLang commit: $(cat /opt/sglang_commit.txt)" && \
    [ "$(cat /opt/sglang_commit.txt)" = "ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb" ] || exit 1

# Set the working directory
WORKDIR /sgl-workspace

# Label for tracking
LABEL org.opencontainers.image.revision="ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.description="SGLang Docker image for commit ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb (2024-04-17)"