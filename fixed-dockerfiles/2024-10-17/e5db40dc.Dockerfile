# Dockerfile for SGLang commit from 2024-10-17
# Base image: nvidia/cuda:12.1.1-devel-ubuntu20.04 (for torch 2.4.x)
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.1

# 1st occurrence of commit SHA (HARDCODED)
ENV SGLANG_COMMIT=e5db40dcbce67157e005f524bf6a5bea7dcb7f34

# Install system dependencies and Python 3.10
RUN apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        curl \
        git \
        wget \
        build-essential \
        ninja-build \
        cmake && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify Python version
RUN python3 --version && python3 -m pip --version

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install torch 2.4.0 with CUDA 12.1 FIRST (from vLLM requirements)
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.5 WITHOUT dependencies to avoid torch version conflicts
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies manually (from vLLM requirements-common.txt and requirements-cuda.txt)
RUN pip install \
    psutil \
    sentencepiece \
    "numpy<2.0.0" \
    requests \
    tqdm \
    py-cpuinfo \
    "transformers>=4.43.2" \
    "tokenizers>=0.19.1" \
    protobuf \
    fastapi \
    aiohttp \
    "openai>=1.0" \
    "uvicorn[standard]" \
    "pydantic>=2.8" \
    pillow \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.10.6" \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    "gguf==0.9.1" \
    importlib_metadata \
    "ray>=2.9" \
    nvidia-ml-py

# Install xformers 0.0.27.post2 (from vLLM requirements-cuda.txt)
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install vllm-flash-attn (from vLLM requirements-cuda.txt)
RUN pip install vllm-flash-attn==2.6.1

# Install flashinfer from wheels (available for torch 2.4 + CUDA 12.1)
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# 2nd occurrence of commit SHA (HARDCODED) - Clone SGLang at exact commit
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout e5db40dcbce67157e005f524bf6a5bea7dcb7f34

# 3rd occurrence of commit SHA (HARDCODED) - Verify commit
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e5db40dcbce67157e005f524bf6a5bea7dcb7f34" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: got $ACTUAL, expected $EXPECTED" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas or malformed lists
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml && \
    sed -i 's/\[\s*\]/[]/g' python/pyproject.toml

# Install additional SGLang dependencies
RUN pip install \
    orjson \
    decord \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    python-multipart \
    torchao \
    uvloop \
    zmq \
    modelscope

# Install SGLang from python/ subdirectory (pyproject.toml is in python/)
RUN cd /sgl-workspace/sglang && \
    pip install -e "python[all]"

# Final verification of all imports
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')" && \
    python3 -c "import flashinfer; print('flashinfer import OK')" && \
    python3 -c "import sglang; print('SGLang import OK')"

# Set environment variable for better performance
ENV NCCL_P2P_DISABLE=1
ENV TORCH_CUDA_ARCH_LIST="9.0"

CMD ["/bin/bash"]