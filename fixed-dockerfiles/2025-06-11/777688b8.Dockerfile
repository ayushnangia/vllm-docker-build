# Dockerfile for SGLang commit from June 11, 2025
# Using June 2024 era compatible versions

# Use PyTorch 2.3.0 base image with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /sgl-workspace

# Build settings for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# 1st SHA occurrence: ENV variable
ENV SGLANG_COMMIT=777688b8929c877e4e28c2eac208d776abe4c3af

# Create constraints file with discovered versions from PyPI (June 2024 era)
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for June 2024 era
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.3
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.0.3
transformers==4.40.0
tokenizers==0.19.1
prometheus_client==0.18.0
prometheus-fastapi-instrumentator==7.0.0
tiktoken==0.6.0
lm-format-enforcer==0.10.1
pillow
aiohttp
openai
psutil
sentencepiece
numpy
requests
py-cpuinfo
filelock==3.10.4
EOF

# Install vLLM 0.5.0 with --no-deps (released June 11, 2024)
RUN pip install vllm==0.5.0 --no-deps

# Install vLLM dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    cmake ninja psutil sentencepiece numpy requests py-cpuinfo \
    transformers tokenizers fastapi aiohttp openai uvicorn pydantic \
    pillow prometheus_client prometheus-fastapi-instrumentator tiktoken \
    lm-format-enforcer outlines typing_extensions filelock

# Install flashinfer from wheels (for torch 2.3, CUDA 12.1)
RUN pip install flashinfer==0.1.0+cu121torch2.3 \
    --find-links https://flashinfer.ai/whl/cu121/torch2.3/flashinfer/

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.7

# 2nd SHA occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 777688b8929c877e4e28c2eac208d776abe4c3af

# 3rd SHA occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="777688b8929c877e4e28c2eac208d776abe4c3af" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Patch pyproject.toml to remove version pins that don't exist in June 2024
RUN cd /sgl-workspace/sglang/python && \
    cp pyproject.toml pyproject.toml.orig && \
    sed -i 's/"torch==2.7.1"/"torch"/g' pyproject.toml && \
    sed -i 's/"torchvision==0.22.1"/"torchvision"/g' pyproject.toml && \
    sed -i 's/"torchaudio==2.7.1"/"torchaudio"/g' pyproject.toml && \
    sed -i 's/"flashinfer_python==0.2.6.post1"/"flashinfer"/g' pyproject.toml && \
    sed -i 's/"transformers==4.52.3"/"transformers>=4.40.0"/g' pyproject.toml && \
    sed -i 's/"torchao==0.9.0",*//g' pyproject.toml && \
    sed -i 's/"xgrammar==0.1.19",*//g' pyproject.toml && \
    sed -i 's/"llguidance>=0.7.11,<0.8.0",*//g' pyproject.toml && \
    sed -i '/^[[:space:]]*,$/d' pyproject.toml && \
    sed -i 's/,]/]/g' pyproject.toml && \
    sed -i 's/,,/,/g' pyproject.toml

# Install SGLang dependencies from pyproject.toml
RUN pip install -c /opt/constraints.txt \
    blobfile==3.0.0 \
    compressed-tensors \
    datasets \
    hf_transfer \
    huggingface_hub \
    interegular \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pynvml \
    python-multipart \
    soundfile==0.13.1 \
    scipy \
    uvloop \
    einops \
    cuda-python

# Install SGLang with --no-deps in editable mode
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install base SGLang dependencies
RUN pip install aiohttp requests tqdm numpy IPython setproctitle

# Set environment variables
ENV PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"

# Final verification
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')" && \
    echo "Build completed successfully for commit: $(cat /opt/sglang_commit.txt)"

# Default command
CMD ["/bin/bash"]