# Dockerfile for SGLang commit (early 2024)
# Date: 2024-02-03
# SGLang version: 0.1.11 (very early commit)
# Architecture: linux/amd64 (GPU)

# Use PyTorch base image for early 2024 (includes Python, CUDA 12.1, PyTorch 2.1.2)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HARDCODED commit SHA (3 occurrences total in this file)
ENV SGLANG_COMMIT=45d6592d4053fe8b2b8dc9440f64c900de040d09

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# The pytorch base image already includes torch 2.1.2, but let's ensure it's the right version
RUN pip show torch | grep Version || pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM first (required by pyproject.toml, version >=0.2.5)
# For this early commit, vLLM 0.2.5 is appropriate
RUN pip install vllm==0.2.5

# Clone SGLang repo and checkout EXACT commit (HARDCODED SHA, occurrence #2)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 45d6592d4053fe8b2b8dc9440f64c900de040d09

# VERIFY the checkout - compare against HARDCODED expected value (occurrence #3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="45d6592d4053fe8b2b8dc9440f64c900de040d09" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source
# Note: For this early commit, the package structure is simpler
WORKDIR /sgl-workspace/sglang
RUN pip install -e "python[all]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')"

# Set working directory
WORKDIR /sgl-workspace

# Default command
CMD ["python3", "-c", "import sglang; print('SGLang loaded successfully')"]