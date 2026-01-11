# SGLang Dockerfile for commit da19434c2f3cbe4f367f84993da0bcbd84efb6ba
# Date: 2024-04-25
# SGLang version: 0.1.14

# Base image for torch 2.1.2 (required by vLLM 0.3.3)
# Using pytorch base image for better compatibility
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=da19434c2f3cbe4f367f84993da0bcbd84efb6ba

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Pre-install torch 2.1.2 with CUDA 12.1 (required by vLLM 0.3.3)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.3.3 with --no-deps to avoid pulling wrong torch version
RUN pip install vllm==0.3.3 --no-deps

# Install vLLM dependencies manually (based on vLLM 0.3.3 requirements.txt)
RUN pip install \
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
    "cupy-cuda12x==12.1.0"

# Clone SGLang repo and checkout the EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout da19434c2f3cbe4f367f84993da0bcbd84efb6ba

# VERIFY the commit matches exactly (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="da19434c2f3cbe4f367f84993da0bcbd84efb6ba" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from the checked-out source
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Install additional packages that might be useful
RUN pip install datasets

# Verify all imports and versions
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; assert torch.__version__.startswith('2.1.2'), f'Wrong torch version: {torch.__version__}'; print(f'Torch version: {torch.__version__}')" && \
    python3 -c "import xformers; print(f'xformers version: {xformers.__version__}')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]