# SGLang Docker image for October 2024 commit
# Date: 2024-10-17
# Based on discovered versions from PyPI for October 17, 2024
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
# 1st HARDCODED occurrence of commit SHA
ENV SGLANG_COMMIT=e5db40dcbce67157e005f524bf6a5bea7dcb7f34
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential \
    libibverbs-dev \
    ninja-build \
    cmake \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA deprecated for Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    cd .. && rm -rf Python-3.10.14*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 with CUDA 12.1 (vLLM 0.5.5 requirement)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with versions discovered from PyPI for 2024-10-17
RUN cat > /opt/constraints.txt <<'EOF'
# Package versions as of October 17, 2024 (discovered from PyPI)
# pydantic 2.9.2 was released Sept 17, 2024 (latest before Oct 17)
pydantic==2.9.2
pydantic-core==2.23.4
# fastapi 0.115.2 was released Oct 12, 2024 (latest before Oct 17)
fastapi==0.115.2
# uvicorn 0.32.0 was released Oct 15, 2024 (latest before Oct 17)
uvicorn==0.32.0
# typing_extensions 4.12.2 was released June 7, 2024 (stable, avoids Sentinel issues)
typing_extensions==4.12.2
# outlines 0.0.46 compatible with pydantic v2
outlines==0.0.46
# pyzmq 26.2.0 was released Aug 22, 2024
pyzmq==26.2.0
# Constrain numpy to avoid v2
numpy<2.0.0
EOF

# Install vLLM 0.5.5 without dependencies first
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies with constraints (from vLLM 0.5.5 requirements)
RUN pip install -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    'numpy<2.0.0' \
    requests \
    tqdm \
    py-cpuinfo \
    'transformers>=4.43.2' \
    'tokenizers>=0.19.1' \
    protobuf \
    fastapi \
    aiohttp \
    'openai>=1.0' \
    'uvicorn[standard]' \
    'pydantic>=2.8' \
    pillow \
    'prometheus-client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' \
    'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.6' \
    'outlines<0.1,>=0.0.43' \
    'typing-extensions>=4.10' \
    'filelock>=3.10.4' \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    'gguf==0.9.1' \
    importlib-metadata \
    'ray>=2.9' \
    nvidia-ml-py

# Install CUDA-specific vLLM dependencies
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install vllm-flash-attn==2.6.1

# Install flashinfer from wheels (available for cu121/torch2.4)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang at exact commit
WORKDIR /sgl-workspace
# 2nd HARDCODED occurrence of commit SHA
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout e5db40dcbce67157e005f524bf6a5bea7dcb7f34

# 3rd HARDCODED occurrence of commit SHA - Verify the commit
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e5db40dcbce67157e005f524bf6a5bea7dcb7f34" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang without dependencies
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# Install SGLang runtime dependencies with constraints (from pyproject.toml)
RUN pip install -c /opt/constraints.txt \
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
    'outlines>=0.0.44' \
    modelscope \
    'openai>=1.0' \
    'tiktoken' \
    'anthropic>=0.20.0' \
    'litellm>=1.0.0'

# Install sgl-kernel (has wheels available on PyPI)
RUN pip install sgl-kernel

# Verify all imports work
# Note: vLLM and outlines imports can fail without GPU - check via pip
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    pip show outlines > /dev/null && echo "Outlines installed OK" && \
    python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import flashinfer; print('flashinfer import successful')"

# Set working directory
WORKDIR /workspace

# Final verification of commit SHA
RUN echo "Commit SHA in /opt/sglang_commit.txt:" && cat /opt/sglang_commit.txt

CMD ["/bin/bash"]