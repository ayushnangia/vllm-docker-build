# Fixed Dockerfile for SGLang commit 09deb20d (2024-05-11)
# Triton Era - uses torch 2.3.0 + flashinfer 0.0.4+ + vLLM 0.4.2

# Base image for torch 2.3.0 with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Pre-install torch 2.3.0 (already in base image, but ensure correct version)
RUN pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers compatible with torch 2.3.0
RUN pip3 install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Build flashinfer from source (version 0.0.4 requirement, build v0.1.2)
RUN pip3 install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.1.2 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . && \
    cd / && rm -rf /tmp/flashinfer

# Install vLLM 0.4.2 with --no-deps to avoid torch version conflicts
RUN pip3 install vllm==0.4.2 --no-deps

# Install vLLM dependencies manually (from requirements-cuda.txt and requirements-common.txt)
# CRITICAL: FastAPI >= 0.126.0 forces pydantic v2, so pin FastAPI < 0.126 for pydantic v1
RUN pip3 install \
    "cmake>=3.21" \
    ninja \
    psutil \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    "transformers>=4.40.0" \
    "tokenizers>=0.19.1" \
    "fastapi<0.126.0" \
    openai \
    "uvicorn[standard]" \
    "pydantic>=1.10,<2.0" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken==0.6.0" \
    "typing_extensions>=4.5.0,<4.12.0" \
    "filelock>=3.10.4" \
    "ray>=2.9" \
    nvidia-ml-py \
    "vllm-nccl-cu12>=2.18,<2.19"

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=09deb20deef8181a23f66c933ea74b86fee47366

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 09deb20deef8181a23f66c933ea74b86fee47366

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="09deb20deef8181a23f66c933ea74b86fee47366" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps (flashinfer, vllm, pydantic)
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"pydantic[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install additional SGLang dependencies
RUN pip3 install \
    aiohttp \
    pyzmq \
    rpyc \
    interegular \
    pillow \
    packaging \
    datasets \
    uvloop

# Install SGLang from source (pyproject.toml is in python/ subdirectory)
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Force pydantic v1 - uninstall any v2 bits that snuck in, reinstall v1
RUN pip3 uninstall -y pydantic pydantic-core 2>/dev/null || true && \
    pip3 install "pydantic>=1.10,<2.0" "typing_extensions>=4.5.0,<4.12.0"

# Replace triton with triton-nightly for better compatibility
RUN pip3 uninstall -y triton triton-nightly || true && \
    pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify all imports work
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import xformers; print(f'xformers: {xformers.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Final verification of commit
RUN echo "=== Final Commit Verification ===" && \
    cat /opt/sglang_commit.txt && \
    echo "=== Installation location ===" && \
    pip3 show sglang | grep -E "^(Name|Version|Location|Editable)"

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]