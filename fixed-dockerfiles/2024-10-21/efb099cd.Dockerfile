# Fixed Dockerfile for SGLang commit efb099cdee90b9ad332fcda96d89dd91ddebe072 (2024-10-21)
# Using torch 2.4.0 with CUDA 12.1 as required by vLLM 0.6.3.post1

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables for build
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st SHA occurrence: ENV
ENV SGLANG_COMMIT=efb099cdee90b9ad332fcda96d89dd91ddebe072

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    vim \
    build-essential \
    cmake \
    ninja-build \
    libibverbs-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip \
    && cd .. && rm -rf Python-3.10.14*

# Verify Python installation
RUN python3 --version && pip3 --version

# Upgrade pip and install base packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 (required by vLLM 0.6.3.post1)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI for October 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-10-21 era
# Core dependencies - CRITICAL: outlines 0.0.44 requires pydantic>=2.0
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
pydantic-core==2.23.4
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
prometheus-client==0.21.0

# Additional dependencies
aiohttp==3.10.5
orjson==3.10.7
packaging==24.1
pillow==10.4.0
psutil==6.0.0
python-multipart==0.0.9
uvloop==0.20.0
huggingface_hub==0.24.7
hf_transfer==0.1.8
interegular==0.3.3
decord==0.6.0
modelscope==1.18.1

# vLLM specific
transformers==4.45.2
tokenizers==0.20.0
numpy<2.0.0
mistral-common==1.4.4
ray==2.37.0
lm-format-enforcer==0.10.6
gguf==0.10.0
compressed-tensors==0.6.0

# For MiniCPM models
datamodel_code_generator==0.25.2
EOF

# Install vLLM 0.6.3.post1 with --no-deps first
RUN pip install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    psutil \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    pynvml \
    triton \
    einops \
    tiktoken \
    msgpack \
    protobuf \
    fastapi \
    uvicorn \
    pydantic \
    aioprometheus \
    aiohttp \
    openai \
    tqdm \
    lm-format-enforcer \
    outlines \
    typing-extensions \
    filelock \
    pyzmq \
    py-cpuinfo \
    transformers \
    tokenizers \
    huggingface-hub \
    numpy \
    nvidia-ml-py \
    ray \
    mistral-common \
    gguf \
    compressed-tensors \
    setuptools \
    grpcio \
    sentencepiece \
    msgpack-numpy \
    partial-json-parser

# Install xformers for Linux x86_64
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer for torch 2.4
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install torchao (no strict version requirement)
RUN pip install torchao --index-url https://download.pytorch.org/whl/cu121

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.3.4

# Work directory
WORKDIR /sgl-workspace

# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout efb099cdee90b9ad332fcda96d89dd91ddebe072 && \
    git submodule update --init --recursive

# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="efb099cdee90b9ad332fcda96d89dd91ddebe072" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps to avoid dependency conflicts
RUN cd /sgl-workspace/sglang && \
    pip install -e python --no-deps

# Install SGLang runtime dependencies with constraints
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
    orjson \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    pyzmq \
    outlines \
    modelscope \
    openai \
    tiktoken \
    anthropic \
    litellm

# Additional dependency for MiniCPM models
RUN pip install datamodel_code_generator

# Replace triton with triton-nightly for better compatibility if needed
RUN pip3 uninstall -y triton triton-nightly 2>/dev/null || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Sanity check: Verify installation
RUN python3 -c "import sglang; print('SGLang version:', sglang.__version__)" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import flashinfer; print('FlashInfer imported successfully')" && \
    python3 -c "import torch; print('PyTorch version:', torch.__version__)" && \
    python3 -c "import pydantic; print('Pydantic version:', pydantic.__version__)"

# Verify commit file exists and contains correct SHA
RUN test -f /opt/sglang_commit.txt && \
    grep -q "efb099cdee90b9ad332fcda96d89dd91ddebe072" /opt/sglang_commit.txt && \
    echo "Commit verification successful: $(cat /opt/sglang_commit.txt)"

# Set working directory to sglang
WORKDIR /sgl-workspace/sglang

# Clean up apt cache and pip cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip3 cache purge

# Set environment back to interactive
ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["/bin/bash"]