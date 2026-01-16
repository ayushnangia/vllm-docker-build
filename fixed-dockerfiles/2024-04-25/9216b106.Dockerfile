# Base image for torch 2.2.1 (required by vLLM 0.4.1)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    numactl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Create workspace
WORKDIR /sgl-workspace

# Set build environment variables
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Create constraints file with discovered versions for April 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-04-25 era
# FastAPI 0.110.2 released April 19, 2024
# uvicorn 0.29.0 released March 20, 2024
# pydantic 2.7.1 released April 23, 2024 (vLLM requires >= 2.0)
# typing_extensions 4.11.0 released April 5, 2024
# outlines 0.0.34 pinned by vLLM 0.4.1
# pyzmq 26.0.2 released April 19, 2024
fastapi==0.110.2
uvicorn==0.29.0
pydantic==2.7.1
typing_extensions==4.11.0
outlines==0.0.34
pyzmq==26.0.2
EOF

# Install vLLM 0.4.1 (released April 24, 2024) with --no-deps to avoid conflicts
RUN pip install vllm==0.4.1 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    'transformers>=4.40.0' \
    'tokenizers>=0.19.1' \
    fastapi \
    'uvicorn[standard]' \
    'pydantic>=2.0' \
    'prometheus_client>=0.18.0' \
    'tiktoken==0.6.0' \
    'lm-format-enforcer==0.9.8' \
    outlines \
    typing_extensions \
    'filelock>=3.10.4' \
    'ray>=2.9' \
    nvidia-ml-py

# Install xformers with --no-deps to prevent pulling wrong torch version
RUN pip install xformers==0.0.25 --no-deps

# Install flashinfer from wheels (available for torch 2.2)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.2/

# 1st SHA occurrence: ENV variable
ENV SGLANG_COMMIT=9216b10678a036a1797e19693b0445c889016687

# Clone and checkout SGLang at specific commit
# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 9216b10678a036a1797e19693b0445c889016687

# 3rd SHA occurrence: Verify commit and save to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9216b10678a036a1797e19693b0445c889016687" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps first
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
# NOTE: Do NOT reinstall torch - keep the 2.2.1 from base image
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    uvloop \
    uvicorn \
    pyzmq \
    interegular \
    pydantic \
    pillow \
    'outlines>=0.0.27' \
    'openai>=1.0' \
    "numpy<2.0" \
    tiktoken \
    'anthropic>=0.20.0'

# Install sgl-kernel if available
RUN pip install sgl-kernel || echo "sgl-kernel not available, skipping"

# Final sanity check - verify imports
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')"

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]