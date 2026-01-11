# Base image with torch 2.1.2 (required by vLLM 0.3.3)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Verify torch version from base image
RUN python3 -c "import torch; assert torch.__version__.startswith('2.1.2'), f'Wrong torch version: {torch.__version__}'"

# Install vLLM 0.3.3 with dependencies (use --no-deps to avoid torch conflicts)
RUN pip install vllm==0.3.3 --no-deps && \
    pip install \
        ninja \
        psutil \
        "ray>=2.9" \
        sentencepiece \
        numpy \
        "transformers>=4.38.0" \
        "xformers==0.0.23.post1" \
        fastapi \
        "uvicorn[standard]" \
        "pydantic>=2.0" \
        "prometheus_client>=0.18.0" \
        "pynvml==11.5.0" \
        "triton>=2.1.0" \
        "outlines>=0.0.27" \
        cupy-cuda12x==12.1.0

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=9216b10678a036a1797e19693b0445c889016687

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 9216b10678a036a1797e19693b0445c889016687

# VERIFY the checkout - compare against HARDCODED expected value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9216b10678a036a1797e19693b0445c889016687" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install additional dependencies from SGLang's pyproject.toml
RUN pip install \
    aiohttp \
    zmq \
    pyzmq \
    rpyc \
    uvloop \
    interegular \
    pillow \
    "openai>=1.0" \
    "anthropic>=0.20.0" \
    tiktoken \
    requests \
    tqdm \
    datasets

# Install SGLang from source
# Note: The pyproject.toml is in the python/ subdirectory as shown in the original Dockerfile
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Final verification - ensure all critical imports work
RUN python3 -c "import sglang; print('SGLang import OK')"
RUN python3 -c "import vllm; print('vLLM import OK')"
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}')"
RUN python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]