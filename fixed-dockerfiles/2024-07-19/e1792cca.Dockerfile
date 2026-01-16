# Base image for torch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Hardcoded commit SHA - 1st occurrence
ENV SGLANG_COMMIT=e1792cca2491af86f29782a3b83533a6566ac75b

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
        git \
        curl \
        wget \
        vim \
        build-essential \
        cmake \
        ninja-build \
        python3-dev \
        python3-pip \
        sudo \
        ccache && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with discovered versions from PyPI for July 19, 2024
# These versions were discovered via WebFetch from PyPI release histories
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-07-19 era
# fastapi 0.111.1 was released July 14, 2024 (latest before July 19)
fastapi==0.111.1
# uvicorn 0.30.1 was released June 2, 2024 (latest before July 19)
uvicorn==0.30.1
# pydantic 2.8.2 was released July 4, 2024 (pydantic v2 required by outlines 0.0.44)
pydantic==2.8.2
# typing_extensions 4.12.2 was released June 7, 2024
typing_extensions==4.12.2
# outlines 0.0.44 was released June 14, 2024 (minimum required by SGLang)
outlines==0.0.44
# pyzmq 26.0.3 was released May 1, 2024
pyzmq==26.0.3
# numpy constraint from vLLM
numpy<2.0.0
EOF

# Install vLLM 0.5.1 with --no-deps
RUN pip install vllm==0.5.1 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.42.0" \
    "tokenizers>=0.19.1" \
    fastapi \
    aiohttp \
    openai \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.1" \
    "outlines>=0.0.43" \
    typing_extensions \
    "filelock>=3.10.4" \
    "ray>=2.9" \
    nvidia-ml-py \
    torchvision==0.18.0 \
    vllm-flash-attn==2.5.9

# Install xformers with --no-deps to prevent pulling wrong torch version
RUN pip install xformers==0.0.26.post1 --no-deps

# Install triton
RUN pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Try to install flashinfer with fallback to building from source
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/ || \
    (echo "flashinfer wheels not found, building from source..." && \
     git clone https://github.com/flashinfer-ai/flashinfer.git && \
     cd flashinfer/python && \
     pip install -e . && \
     cd /sgl-workspace && \
     rm -rf flashinfer)

# Hardcoded commit SHA - 2nd occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout e1792cca2491af86f29782a3b83533a6566ac75b

# Hardcoded commit SHA - 3rd occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e1792cca2491af86f29782a3b83533a6566ac75b" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Build and install sgl-kernel from source (if it exists)
RUN if [ -d "/sgl-workspace/sglang/python/sglang/srt/sgl_kernel" ] && [ -f "/sgl-workspace/sglang/python/sglang/srt/sgl_kernel/setup.py" ]; then \
    cd /sgl-workspace/sglang/python/sglang/srt/sgl_kernel && pip install -e .; \
    else echo "sgl-kernel not found in this commit, skipping"; fi

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
# Note: Don't reinstall torch - keep the one from base image
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    pydantic \
    uvicorn \
    uvloop \
    pyzmq \
    "openai>=1.0" \
    tiktoken \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Additional dependencies from outlines 0.0.44 (discovered via WebFetch from GitHub)
RUN pip install -c /opt/constraints.txt \
    interegular \
    jinja2 \
    lark \
    nest_asyncio \
    cloudpickle \
    diskcache \
    numba \
    referencing \
    jsonschema \
    datasets \
    pycountry \
    pyairports

# Cleanup
RUN pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set interactive mode back
ENV DEBIAN_FRONTEND=interactive

WORKDIR /sgl-workspace

# Verification commands
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
# Note: torch.cuda.is_available() returns False without GPU - skip this check
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"