# Fixed Dockerfile for SGLang (2024-04-17)
# Base image for torch 2.1.2 with CUDA 12.1
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /sgl-workspace

# Set environment variables
ENV SGLANG_COMMIT=e822e5900b98d89d19e0a293d9ad384f4df2945a \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=96

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create constraints file with versions discovered from PyPI for 2024-04-17 era
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions from PyPI for April 17, 2024
fastapi==0.110.1
uvicorn==0.29.0
pydantic==2.7.0
typing_extensions==4.11.0
outlines==0.0.39
pyzmq==26.0.0
aiohttp==3.9.5
interegular==0.3.3
# vLLM 0.4.0.post1 requirements
transformers==4.39.3
xformers==0.0.23.post1
triton==2.1.0
tiktoken==0.6.0
sentencepiece
ray>=2.9
py-cpuinfo
psutil
numpy
requests
pynvml==11.5.0
prometheus_client>=0.18.0
# SGLang extra dependencies
uvloop
rpyc
pillow
openai>=1.0
anthropic>=0.20.0
tqdm
datasets
EOF

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install torch explicitly (matching vLLM requirement)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.4.0.post1 with --no-deps first
RUN pip install vllm==0.4.0.post1 --no-deps

# Install vLLM dependencies using constraints where applicable
RUN pip install -c /opt/constraints.txt \
    cmake>=3.21 \
    ninja \
    psutil \
    "ray>=2.9" \
    sentencepiece \
    numpy \
    requests \
    py-cpuinfo \
    "transformers>=4.39.1" \
    xformers==0.0.23.post1 \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "prometheus_client>=0.18.0" \
    pynvml==11.5.0 \
    "triton>=2.1.0" \
    outlines==0.0.34 \
    tiktoken==0.6.0

# Clone SGLang at specific commit (2nd SHA occurrence)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout e822e5900b98d89d19e0a293d9ad384f4df2945a

# Verify commit SHA and save to file (3rd SHA occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e822e5900b98d89d19e0a293d9ad384f4df2945a" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Build and install flashinfer from source (required for this era)
RUN cd /sgl-workspace && \
    git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.0.2 && \
    cd python && \
    pip install . --no-deps && \
    cd /sgl-workspace && \
    rm -rf flashinfer

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies from pyproject.toml using constraints
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    aiohttp \
    fastapi \
    psutil \
    rpyc \
    torch \
    uvloop \
    uvicorn \
    pyzmq \
    "vllm>=0.3.3" \
    interegular \
    pydantic \
    pillow \
    "outlines>=0.0.27" \
    "openai>=1.0" \
    numpy \
    tiktoken \
    "anthropic>=0.20.0" \
    datasets

# Update to use correct outlines version from constraints (0.0.39 for this era)
RUN pip install --force-reinstall -c /opt/constraints.txt outlines==0.0.39

# Final verification
RUN python -c "import sglang; print('SGLang imported successfully')" && \
    python -c "import vllm; print('vLLM imported successfully')" && \
    python -c "import outlines; print('Outlines imported successfully')" && \
    python -c "import flashinfer; print('Flashinfer imported successfully')" && \
    python -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Set the working directory
WORKDIR /sgl-workspace

# Set default command
CMD ["/bin/bash"]