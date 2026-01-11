# Dockerfile for SGLang commit e822e5900b98d89d19e0a293d9ad384f4df2945a (2024-04-17)
# Based on discovered dependencies:
# - vLLM 0.3.3 requires torch 2.1.2, xformers 0.0.23.post1
# - pyproject.toml is in python/ subdirectory

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV SGLANG_COMMIT=e822e5900b98d89d19e0a293d9ad384f4df2945a
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /sgl-workspace

# Install torch explicitly first (matching vLLM requirement)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.3.3 without dependencies to control versions
RUN pip install --no-cache-dir vllm==0.3.3 --no-deps

# Install vLLM dependencies based on vllm v0.3.3 requirements.txt
RUN pip install --no-cache-dir \
    ninja \
    psutil \
    ray>=2.9 \
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

# Clone SGLang repository at exact commit (2nd occurrence of SHA)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout e822e5900b98d89d19e0a293d9ad384f4df2945a

# Patch pyproject.toml to remove vllm since we installed it manually
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang with all optional dependencies
# pyproject.toml is in python/ subdirectory
RUN cd /sgl-workspace/sglang && \
    pip install --no-cache-dir -e "python[all]"

# Install additional useful packages
RUN pip install --no-cache-dir \
    datasets \
    jupyter \
    ipywidgets

# Verify commit SHA (3rd occurrence of SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e822e5900b98d89d19e0a293d9ad384f4df2945a" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: Expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Final verification of imports
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')" && \
    python3 -c "import transformers; print('transformers import OK')"

# Set default working directory
WORKDIR /sgl-workspace

# Entry point
CMD ["/bin/bash"]