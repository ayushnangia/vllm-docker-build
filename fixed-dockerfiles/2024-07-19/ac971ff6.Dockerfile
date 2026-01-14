# SGLang Docker for commit ac971ff633de330de3ded7f7475caaf7cd5bbdcd
# Date: 2024-07-19
# SGLang v0.1.21, vLLM v0.5.1, torch 2.3.0
# Discovered versions from PyPI research

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# HARDCODE 1st occurrence of SHA
ENV SGLANG_COMMIT=ac971ff633de330de3ded7f7475caaf7cd5bbdcd

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
        git \
        curl \
        sudo \
        build-essential \
        python3-dev \
        ccache \
        cmake \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Create constraints file with versions discovered from PyPI for July 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Discovered from PyPI for 2024-07-19 era
# pydantic 2.8.2 released July 4, 2024 (needs pydantic v2 for outlines>=0.0.44)
pydantic==2.8.2
# typing_extensions 4.12.2 released June 7, 2024 (avoids Sentinel issue in 4.14+)
typing_extensions==4.12.2
# fastapi 0.111.1 released July 14, 2024
fastapi==0.111.1
# uvicorn 0.30.3 released July 2024
uvicorn==0.30.3
# outlines 0.0.46 released June 22, 2024 (requires pydantic>=2.0)
outlines==0.0.46
# pyzmq 26.0.3 released May 2024
pyzmq==26.0.3
# aiohttp 3.9.5 released April 2024
aiohttp==3.9.5
# pydantic-core 2.20.1 released July 2024 (for pydantic 2.8.2)
pydantic-core==2.20.1
EOF

# Install vLLM 0.5.1 with --no-deps to avoid dependency conflicts
RUN pip install --no-cache-dir vllm==0.5.1 --no-deps

# Install vLLM dependencies from requirements-common.txt with constraints
RUN pip install --no-cache-dir \
    -c /opt/constraints.txt \
    cmake>=3.21 \
    ninja \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.42.0" \
    "tokenizers>=0.19.1" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.1" \
    "filelock>=3.10.4" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    pillow \
    openai

# Install vLLM CUDA dependencies from requirements-cuda.txt
RUN pip install --no-cache-dir \
    "ray>=2.9" \
    nvidia-ml-py \
    torchvision==0.18.0 \
    xformers==0.0.26.post1 \
    vllm-flash-attn==2.5.9

# Build flashinfer from source (no prebuilt wheels for torch 2.3 at cu121)
RUN cd /tmp && \
    git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.0.8 && \
    cd python && \
    pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/flashinfer

# Clone SGLang at specific commit
WORKDIR /sgl-workspace
# HARDCODE 2nd occurrence of SHA
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout ac971ff633de330de3ded7f7475caaf7cd5bbdcd

# Verify commit SHA - HARDCODE 3rd occurrence of SHA
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ac971ff633de330de3ded7f7475caaf7cd5bbdcd" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected: $EXPECTED, Got: $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang with --no-deps (from python subdirectory)
RUN cd /sgl-workspace/sglang/python && \
    pip install --no-cache-dir -e . --no-deps

# Install SGLang dependencies from pyproject.toml[srt] with constraints
RUN pip install --no-cache-dir \
    -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    interegular \
    packaging \
    hf_transfer \
    huggingface_hub \
    uvloop \
    pyzmq

# Install SGLang optional dependencies
RUN pip install --no-cache-dir \
    "openai>=1.0" \
    tiktoken \
    "anthropic>=0.20.0" \
    "litellm>=1.0.0"

# Install sgl-kernel from PyPI (version 0.1.9 available in July 2024)
RUN pip install --no-cache-dir sgl-kernel==0.1.9

# Install triton nightly (as per many SGLang Dockerfiles)
RUN pip uninstall -y triton triton-nightly || true && \
    pip install --no-cache-dir --no-deps \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ \
    triton-nightly

# Sanity checks
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}')"
RUN python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
RUN python3 -c "import sglang; print(f'SGLang loaded successfully')"
RUN python3 -c "import flashinfer; print(f'Flashinfer loaded successfully')"
RUN python3 -c "import outlines; print(f'Outlines loaded successfully')"
RUN python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"
RUN python3 -c "import typing_extensions; print(f'typing_extensions loaded successfully')"

# Verify SGLang installation
RUN pip show sglang | grep -E "^(Version|Location|Editable)" || true

# Final verification of commit file
RUN test -f /opt/sglang_commit.txt && \
    echo "Commit proof file exists: $(cat /opt/sglang_commit.txt)"

# Set working directory
WORKDIR /sgl-workspace

# Set entrypoint
ENTRYPOINT ["/bin/bash"]