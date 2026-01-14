FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set build environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# First SHA occurrence: Set commit as ENV
ENV SGLANG_COMMIT=7ce36068914503c3a53ad7be23ab29831fb8aa63

# Install system dependencies and Python 3.10
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y \
        git curl wget build-essential \
        zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
        libibverbs-dev cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
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

# Upgrade pip and essential build tools
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch 2.4.0 for CUDA 12.1
RUN pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from October 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Version constraints discovered from PyPI for 2024-10-21 era
fastapi==0.115.2
uvicorn==0.32.0
pydantic==2.9.2
pydantic-core==2.23.4
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0
# Additional constraints for vLLM dependencies
numpy<2.0.0
transformers==4.45.2
tokenizers==0.20.1
aiohttp==3.10.10
openai==1.51.2
prometheus_client==0.21.0
prometheus-fastapi-instrumentator==7.0.0
tiktoken==0.8.0
lm-format-enforcer==0.10.6
filelock==3.16.1
msgspec==0.18.6
mistral_common==1.4.4
compressed-tensors==0.6.0
EOF

# Install vLLM 0.6.3.post1 without dependencies
RUN pip3 install vllm==0.6.3.post1 --no-deps

# Install vLLM dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    psutil sentencepiece requests tqdm py-cpuinfo protobuf \
    pillow partial-json-parser gguf importlib_metadata \
    pyyaml six einops ray nvidia-ml-py xformers==0.0.27.post2

# Install remaining vLLM dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    numpy transformers tokenizers fastapi aiohttp openai uvicorn[standard] \
    pydantic prometheus_client prometheus-fastapi-instrumentator tiktoken \
    lm-format-enforcer outlines typing_extensions filelock pyzmq msgspec \
    mistral_common[opencv] compressed-tensors

# Install flashinfer from the special repository
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang repository and checkout specific commit (second SHA occurrence)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 7ce36068914503c3a53ad7be23ab29831fb8aa63

# Verify commit and write to file (third SHA occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="7ce36068914503c3a53ad7be23ab29831fb8aa63" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip3 install -e . --no-deps

# Install SGLang dependencies with constraints
RUN pip3 install -c /opt/constraints.txt \
    requests tqdm numpy aiohttp decord hf_transfer huggingface_hub \
    interegular orjson packaging pillow psutil pydantic python-multipart \
    torchao uvicorn uvloop zmq modelscope

# Install additional packages for openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator

# Try to install sgl-kernel if available for performance
RUN pip3 install sgl-kernel || echo "sgl-kernel not available, continuing without it"

# Clean pip cache
RUN pip3 cache purge

# Sanity check: verify core imports work
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import fastapi; print(f'FastAPI imported successfully')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Final environment setup
ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["/bin/bash"]