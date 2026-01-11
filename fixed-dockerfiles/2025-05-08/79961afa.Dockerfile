# Fixed Dockerfile for SGLang
# Date: 2025-05-08
# Built with torch 2.6.0 and CUDA 12.4
# All versions discovered via WebFetch from PyPI

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git curl wget vim \
    build-essential cmake ninja-build \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libibverbs-dev software-properties-common sudo \
    rdma-core infiniband-diags openssh-server perftest \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set build environment
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 1st hardcoded SHA: Set commit SHA as environment variable
ENV SGLANG_COMMIT=79961afa8281f98f380d11db45c8d4b6e66a574f

# Install PyTorch 2.6.0 with CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions from May 2024 era
# All these versions were discovered via WebFetch from PyPI:
# - outlines 0.0.44 requires pydantic >= 2.0 (discovered via wheel metadata extraction)
# - pydantic 2.7.2 from May 28, 2024 (discovered via PyPI history)
# - fastapi 0.111.0 from May 3, 2024 (discovered via PyPI history)
# - typing_extensions 4.11.0 from April 5, 2024 (discovered via PyPI history)
# - uvicorn 0.30.0 from May 28, 2024 (discovered via PyPI history)
# - pyzmq 26.0.3 from May 1, 2024 (discovered via PyPI history)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-05-08 era
fastapi==0.111.0
uvicorn==0.30.0
pydantic==2.7.2
typing_extensions==4.11.0
outlines==0.0.44
pyzmq==26.0.3
transformers==4.51.1
prometheus-client==0.20.0
EOF

# Install vLLM 0.4.2 (from May 5, 2024, discovered via PyPI history) dependencies first
RUN pip install --no-deps vllm==0.4.2
RUN pip install -c /opt/constraints.txt \
    numpy \
    requests \
    psutil \
    sentencepiece \
    py-cpuinfo \
    tokenizers>=0.19.1 \
    tiktoken==0.6.0 \
    lm-format-enforcer==0.9.8 \
    filelock>=3.10.4 \
    nvidia-ml-py \
    prometheus-fastapi-instrumentator>=7.0.0

# Install runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    compressed-tensors \
    datasets \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.11 \
    modelscope \
    ninja \
    orjson \
    packaging \
    pillow \
    pynvml \
    python-multipart \
    soundfile==0.13.1 \
    torchao>=0.9.0 \
    uvloop \
    xgrammar==0.1.17 \
    blobfile==3.0.0

# Install sgl-kernel from PyPI (version 0.1.1 discovered via WebFetch)
RUN pip install sgl-kernel==0.1.1

# Install flashinfer_python with specific CUDA/torch version
# Version 0.2.5 discovered via WebFetch from flashinfer.ai/whl/cu124/torch2.6/
RUN pip install flashinfer_python==0.2.5 --find-links https://flashinfer.ai/whl/cu124/torch2.6/

# Install other SGLang dependencies
RUN pip install \
    cuda-python \
    partial_json_parser \
    einops \
    aiohttp \
    tqdm \
    IPython \
    setproctitle

# For openbmb/MiniCPM models
RUN pip install datamodel_code_generator

# Clone and install SGLang
WORKDIR /sgl-workspace

# 2nd hardcoded SHA: Clone and checkout specific commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 79961afa8281f98f380d11db45c8d4b6e66a574f

# 3rd hardcoded SHA: Verify commit and write to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="79961afa8281f98f380d11db45c8d4b6e66a574f" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL" >&2; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified commit: $ACTUAL"

# Install SGLang in editable mode without dependencies
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Replace system Triton with nightly (common pattern in SGLang Dockerfiles)
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Sanity check: Verify imports work
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import successful')" && \
    python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import outlines; print('Outlines import successful')" && \
    python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import fastapi; print(f'FastAPI import successful')" && \
    python3 -c "import flashinfer; print('FlashInfer import successful')" && \
    python3 -c "import sgl_kernel; print('sgl-kernel import successful')" && \
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Final verification of commit proof file
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists at /opt/sglang_commit.txt" && \
    cat /opt/sglang_commit.txt

# Set working directory
WORKDIR /workspace

ENV DEBIAN_FRONTEND=interactive

# Default command
CMD ["/bin/bash"]