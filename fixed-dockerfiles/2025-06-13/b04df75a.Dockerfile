# Base image for torch 2.7.1 - requires CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 1st commit SHA reference: ENV
ENV SGLANG_COMMIT=b04df75acdda5b99999c02820e64b5b005c07159
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl vim \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up-to-date
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Set Python 3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# Build settings for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install PyTorch 2.7.1 with CUDA 12.4
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions for June 13, 2025
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-06-13 era
fastapi==0.115.12
uvicorn==0.34.3
pydantic==2.11.6
typing_extensions==4.13.2
outlines==0.0.44
pyzmq==27.0.0
transformers==4.52.3
EOF

# Install vLLM 0.9.1 (June 10, 2025) without dependencies
RUN pip install vllm==0.9.1 --no-deps

# Install vLLM's core dependencies
RUN pip install -c /opt/constraints.txt \
    numpy \
    msgspec \
    tqdm \
    requests \
    pyyaml \
    jsonschema \
    pillow \
    prometheus_client \
    tiktoken \
    py-cpuinfo \
    pynvml \
    triton \
    einops \
    xformers \
    filelock \
    pyarrow \
    ray \
    sentencepiece \
    lm-format-enforcer \
    gguf \
    mistral-common \
    compressed-tensors \
    depyf

# Working directory
WORKDIR /sgl-workspace

# 2nd commit SHA reference: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout b04df75acdda5b99999c02820e64b5b005c07159

# 3rd commit SHA reference: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="b04df75acdda5b99999c02820e64b5b005c07159" && \
    test "$ACTUAL" = "$EXPECTED" || exit 1 && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install flashinfer
RUN pip install flashinfer_python==0.2.6.post1 -i https://flashinfer.ai/whl/cu124/torch2.7/

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.8.post1

# Install SGLang in editable mode without dependencies
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# Install SGLang's runtime_common dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.11 \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client==0.20.0 \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq==27.0.0 \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    transformers==4.52.3 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19 \
    cuda-python \
    outlines==0.0.44 \
    einops

# Install vllm-nccl-cu12 for compatibility
RUN pip install vllm-nccl-cu12==2.24.1.3.0

# Sanity check: Verify the installation
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Verify commit SHA is written correctly
RUN test -f /opt/sglang_commit.txt && \
    STORED=$(cat /opt/sglang_commit.txt) && \
    test "$STORED" = "b04df75acdda5b99999c02820e64b5b005c07159" && \
    echo "Commit verification passed: $STORED"

# Set environment variables for runtime
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH

WORKDIR /workspace