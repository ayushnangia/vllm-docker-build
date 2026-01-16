# Fixed Dockerfile for SGLang commit 42a2d82ba71dc86ca3b6342c978db450658b750c
# Date: 2024-09-23
# Using torch 2.4.0 which requires CUDA 12.1 base image

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
# 1st SHA occurrence - ENV variable
ENV SGLANG_COMMIT=42a2d82ba71dc86ca3b6342c978db450658b750c
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies and Python 3.10 from source
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        software-properties-common \
        build-essential \
        curl \
        wget \
        git \
        sudo \
        libibverbs-dev \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        liblzma-dev \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Build Python 3.10 from source (deadsnakes PPA deprecated)
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xzf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/Python-3.10.14* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/local/bin/python3.10 \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && python3 --version \
    && python3 -m pip --version

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.4.0 and related packages
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu121

# Create constraints file based on versions discovered from PyPI for September 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-09-23 era
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
aiohttp==3.10.5
interegular==0.3.3
huggingface_hub==0.24.6
hf_transfer==0.1.8
pillow==10.4.0
psutil==6.0.0
packaging==24.1
python-multipart==0.0.9
decord==0.6.0
uvloop==0.20.0
EOF

# Install vLLM 0.5.5 without dependencies first
RUN python3 -m pip install --no-cache-dir vllm==0.5.5 --no-deps

# Install vLLM dependencies with constraints
RUN python3 -m pip install --no-cache-dir -c /opt/constraints.txt \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    protobuf \
    fastapi \
    aiohttp \
    "openai>=1.0" \
    "uvicorn[standard]" \
    "pydantic>=2.8" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    "gguf==0.9.1" \
    importlib_metadata \
    ray \
    nvidia-ml-py \
    "vllm-flash-attn==2.6.1"

# Install flashinfer from pre-built wheels
RUN python3 -m pip install --no-cache-dir \
    flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang at specific commit
# 2nd SHA occurrence - git checkout
RUN git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang \
    && cd /sgl-workspace/sglang \
    && git checkout 42a2d82ba71dc86ca3b6342c978db450658b750c

# Verify commit SHA and write to file
# 3rd SHA occurrence - verification
RUN cd /sgl-workspace/sglang \
    && ACTUAL=$(git rev-parse HEAD) \
    && EXPECTED="42a2d82ba71dc86ca3b6342c978db450658b750c" \
    && if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi \
    && echo "$ACTUAL" > /opt/sglang_commit.txt \
    && echo "Verified commit: $ACTUAL"

# Install sgl-kernel from PyPI if available, otherwise skip (old commit may not need it)
RUN pip install sgl-kernel==0.3.4 || pip install sgl-kernel || echo "sgl-kernel not available, skipping"

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python \
    && python3 -m pip install --no-cache-dir -e . --no-deps

# Install SGLang dependencies with constraints
RUN python3 -m pip install --no-cache-dir -c /opt/constraints.txt \
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
    torch \
    torchao \
    uvicorn \
    uvloop \
    pyzmq \
    outlines

# Install additional packages for MiniCPM models
RUN python3 -m pip install --no-cache-dir datamodel_code_generator

# Verify installation
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
RUN python3 -c "import sglang; print('SGLang installed successfully')" \
    && pip show vllm > /dev/null && echo "vLLM installed OK" \
    && python3 -c "import torch; print(f'PyTorch {torch.__version__}')" \
    && python3 -c "import flashinfer; print('flashinfer OK')"

# Clean up pip cache
RUN python3 -m pip cache purge

# Reset to interactive mode
ENV DEBIAN_FRONTEND=interactive

# Set entrypoint
WORKDIR /sgl-workspace
CMD ["/bin/bash"]