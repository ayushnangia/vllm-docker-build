# SGLang Dockerfile for commit 6fc175968c3a9fc0521948aa3636887cd6d84107 (2025-05-01)
# Using nvidia/cuda:12.4.1 for torch 2.6.0 compatibility
# All dependency versions discovered from PyPI for May 2025 era

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st SHA occurrence: ENV declaration
ENV SGLANG_COMMIT=6fc175968c3a9fc0521948aa3636887cd6d84107

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip \
    git curl wget build-essential \
    ninja-build cmake \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.6.0 with CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with versions discovered from PyPI for May 2025
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-05-01 era
# Critical: outlines 0.1.11 (Dec 13, 2024) requires pydantic v2
# Critical: typing_extensions must be < 4.14.0 to avoid Sentinel issue
fastapi==0.115.12
uvicorn==0.34.2
pydantic==2.11.4
pydantic-core==2.18.2
typing_extensions==4.13.2
outlines==0.1.11
outlines-core==0.1.26
pyzmq==26.4.0
transformers==4.51.1
flashinfer-python==0.2.5
sgl-kernel==0.1.1
# Additional runtime_common dependencies
aiohttp==3.11.10
requests==2.32.3
tqdm==4.67.1
numpy==2.0.2
setproctitle==1.3.4
huggingface_hub==0.26.5
orjson==3.10.12
packaging==24.2
pillow==11.0.0
prometheus-client==0.21.1
psutil==6.1.1
uvloop==0.21.0
xgrammar==0.1.17
blobfile==3.0.0
interegular==0.3.4
llguidance==0.7.11
ninja==1.11.1.2
hf_transfer==0.1.9
python-multipart==0.0.18
soundfile==0.13.1
compressed-tensors==0.8.1
modelscope==1.20.1
torchao==0.9.0
partial_json_parser==0.3.0
einops==0.8.0
cuda-python==12.6.1
pynvml==11.5.3
decord==0.6.0
datasets==3.2.0
EOF

# Create working directory
RUN mkdir -p /sgl-workspace
WORKDIR /sgl-workspace

# 2nd SHA occurrence: git clone and checkout
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 6fc175968c3a9fc0521948aa3636887cd6d84107

# 3rd SHA occurrence: verification and write to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL_SHA=$(git rev-parse HEAD) && \
    EXPECTED_SHA="6fc175968c3a9fc0521948aa3636887cd6d84107" && \
    if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then \
        echo "ERROR: SHA mismatch! Expected: $EXPECTED_SHA, Got: $ACTUAL_SHA" && exit 1; \
    fi && \
    echo "$ACTUAL_SHA" > /opt/sglang_commit.txt && \
    echo "SHA verification successful: $ACTUAL_SHA"

# Install flashinfer-python from flashinfer.ai wheel index (exact version)
RUN pip install flashinfer-python==0.2.5 -f https://flashinfer.ai/whl/cu124/torch2.6/flashinfer_python

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.1

# Install runtime_common dependencies with constraints
# Using constraints file to ensure compatible versions
RUN pip install -c /opt/constraints.txt \
    compressed-tensors \
    datasets \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance \
    modelscope \
    ninja \
    orjson \
    packaging \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    soundfile \
    torchao \
    transformers \
    uvicorn \
    uvloop \
    xgrammar \
    blobfile

# Install outlines and other SGLang-specific dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    outlines \
    outlines-core \
    partial_json_parser \
    einops \
    cuda-python

# Install additional common Python packages with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle

# Install datamodel_code_generator for MiniCPM models
RUN pip install datamodel_code_generator

# Remove conflicting dependencies from pyproject.toml before installing SGLang
# This prevents pip from trying to reinstall different versions
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchvision[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"cuda-python[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"outlines[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"partial_json_parser[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"einops[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"transformers[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"xgrammar[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"blobfile[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"pyzmq[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"prometheus-client[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"soundfile[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"llguidance[^"]*",*//g' python/pyproject.toml

# Install SGLang from source (editable install)
WORKDIR /sgl-workspace/sglang
RUN pip install -e python

# Install Triton nightly (common for SGLang)
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Sanity checks
RUN python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import flashinfer; print('Flashinfer import successful')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import successful')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import outlines; print('Outlines import successful')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

# Final SHA verification
RUN test "$(cat /opt/sglang_commit.txt)" = "6fc175968c3a9fc0521948aa3636887cd6d84107" || \
    (echo "SHA verification failed!" && exit 1)

# Set working directory
WORKDIR /sgl-workspace/sglang

# Reset frontend
ENV DEBIAN_FRONTEND=interactive

# Clean up
RUN pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Entry point
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]