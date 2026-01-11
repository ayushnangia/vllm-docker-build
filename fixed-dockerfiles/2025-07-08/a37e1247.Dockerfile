# SGLang Dockerfile for commit a37e1247c183cff86a18f2ed1a075e40704b1c5e (2025-07-08)
# Base image for torch 2.7.1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=96

# 1st SHA occurrence: ENV variable
ENV SGLANG_COMMIT=a37e1247c183cff86a18f2ed1a075e40704b1c5e

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git curl wget build-essential cmake ninja-build \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev \
    ccache && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.7.1 with CUDA 12.4
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-07-08 era
fastapi==0.116.0
uvicorn==0.34.3
pydantic==2.11.7
pydantic-core==2.20.1
typing_extensions==4.13.2
outlines==0.1.11
outlines-core==0.1.26
pyzmq==27.0.0
orjson==3.10.6
msgspec==0.18.6
numpy==1.26.4
pillow==10.4.0
prometheus-client==0.20.0
psutil==6.0.0
aiohttp==3.10.0
requests==2.32.3
tqdm==4.66.4
huggingface-hub==0.24.5
datasets==2.20.0
tokenizers==0.19.1
safetensors==0.4.3
accelerate==0.33.0
transformers==4.53.0
blobfile==3.0.0
interegular==0.3.3
jinja2==3.1.4
lark==1.2.2
nest-asyncio==1.6.0
cloudpickle==3.0.0
diskcache==5.6.3
referencing==0.35.1
jsonschema==4.23.0
pycountry==24.6.1
airportsdata==20241001
partial-json-parser==0.2.1.1
setproctitle==1.3.3
hf_transfer==0.1.8
modelscope==1.16.1
packaging==24.1
llguidance==0.7.11
xgrammar==0.1.19
timm==1.0.16
soundfile==0.13.1
scipy==1.14.0
torchao==0.9.0
pybase64==1.4.0
python-multipart==0.0.9
uvloop==0.19.0
einops==0.8.0
compressed-tensors==0.6.0
EOF

# Clone and build flashinfer from source (no wheels for torch 2.7)
WORKDIR /tmp
RUN git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.7 && \
    cd python && \
    pip install ninja && \
    pip install . --no-deps && \
    cd / && rm -rf /tmp/flashinfer

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.2.4 --no-deps

# Clone SGLang repository
WORKDIR /sgl-workspace
# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout a37e1247c183cff86a18f2ed1a075e40704b1c5e

# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="a37e1247c183cff86a18f2ed1a075e40704b1c5e" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install core dependencies first
RUN pip install -c /opt/constraints.txt \
    numpy scipy einops \
    pydantic pydantic-core typing_extensions \
    fastapi uvicorn uvloop orjson msgspec python-multipart \
    aiohttp requests tqdm pyzmq \
    packaging setuptools wheel

# Install ML dependencies
RUN pip install -c /opt/constraints.txt \
    huggingface-hub datasets tokenizers safetensors accelerate \
    transformers timm torchao compressed-tensors

# Install SGLang dependencies from pyproject.toml
RUN pip install -c /opt/constraints.txt \
    blobfile hf_transfer modelscope \
    interegular llguidance ninja \
    outlines outlines-core \
    partial-json-parser pillow prometheus-client psutil \
    pybase64 soundfile xgrammar \
    nest-asyncio cloudpickle diskcache \
    referencing jsonschema pycountry airportsdata \
    lark jinja2 setproctitle IPython build

# Install SGLang in editable mode without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install cuda-python
RUN pip install cuda-python

# Replace Triton with nightly (common pattern in SGLang)
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import flashinfer; print('FlashInfer imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default command
CMD ["/bin/bash"]