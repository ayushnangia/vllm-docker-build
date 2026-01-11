FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /sgl-workspace

# Set the commit SHA as environment variable (1st occurrence)
ENV SGLANG_COMMIT=ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb

# Install torch 2.1.2 first (from vLLM requirements)
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.3.3 without dependencies to avoid pulling wrong torch
RUN pip install vllm==0.3.3 --no-deps

# Install vLLM dependencies manually (from vLLM 0.3.3 requirements.txt)
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

# Clone SGLang at the exact commit (2nd occurrence)
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb

# Verify commit SHA (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: Expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove vllm from dependencies since we already installed it
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang (pyproject.toml is in python/ subdirectory)
RUN cd /sgl-workspace/sglang && \
    pip install -e "python[all]"

# Install additional useful packages
RUN pip install datasets

# Verify installation
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import xformers; print('xformers import OK')"

# Final commit verification
RUN test -f /opt/sglang_commit.txt && \
    echo "SGLang commit: $(cat /opt/sglang_commit.txt)"

WORKDIR /sgl-workspace