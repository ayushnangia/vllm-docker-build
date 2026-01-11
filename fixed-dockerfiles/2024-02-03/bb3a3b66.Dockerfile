# Dockerfile for SGLang commit bb3a3b6675b1844a13ebe368ad693f3dc75b315b
# Date: February 3, 2024
# Architecture: linux/amd64 (GPU)
# Based on pyproject.toml analysis: No flashinfer, vLLM>=0.2.5, torch unspecified

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Torch 2.1.2 is already installed in the pytorch base image
# Verify torch installation and CUDA availability
RUN python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install vLLM 0.2.5 (specified in pyproject.toml as >=0.2.5, using exact version for reproducibility)
RUN pip install vllm==0.2.5

# HARDCODE the commit SHA (1 of 3 occurrences)
ENV SGLANG_COMMIT=bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# Clone SGLang and checkout EXACT commit (2 of 3 occurrences - hardcoded in checkout command)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout bb3a3b6675b1844a13ebe368ad693f3dc75b315b

# VERIFY commit - compare against HARDCODED value (3 of 3 occurrences)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="bb3a3b6675b1844a13ebe368ad693f3dc75b315b" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang package
# Note: pyproject.toml is at the root level for this early commit, not in python/ subdirectory
WORKDIR /sgl-workspace/sglang
RUN pip install -e ".[all]"

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')"

# Additional verification for dependencies
RUN python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import torch; print(f'Torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('All imports successful')"

# Set working directory
WORKDIR /sgl-workspace

# Default command to verify the installation
CMD ["python3", "-c", "import sglang; print('SGLang loaded successfully')"]