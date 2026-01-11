# Fixed Dockerfile for SGLang commit 9d5fa68b903d295d2b39201d54905c6801f60f7f (2025-06-09)
# Using CUDA 12.4 for torch 2.6.0 compatibility
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Build arguments
ARG SGLANG_COMMIT=9d5fa68b903d295d2b39201d54905c6801f60f7f
ENV SGLANG_COMMIT=9d5fa68b903d295d2b39201d54905c6801f60f7f
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.10 python3.10-dev python3-pip \
        git curl wget vim \
        build-essential cmake ninja-build \
        libssl-dev libffi-dev \
        libibverbs-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.6.0 with CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Create constraints file with discovered versions for June 9, 2025
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-06-09 era
fastapi==0.115.11
uvicorn==0.34.3
pydantic==2.11.5
pydantic-core==2.20.1
typing_extensions==4.14.0
pyzmq==26.2.1
outlines==0.1.11
uvloop==0.20.0
httptools==0.6.4
websockets==14.1
watchfiles==1.1.0
python-multipart==0.0.15
email-validator==2.2.0
httpx==0.28.1
pyyaml==6.0.2
ujson==5.10.0
orjson==3.10.11
python-dotenv==1.0.1
itsdangerous==2.2.0
jinja2==3.1.5
python-dateutil==2.9.0.post0
six==1.17.0
EOF

# Build environment variables for compilation
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install vLLM 0.9.0.1 (latest before June 9, 2025) with --no-deps
RUN pip install vllm==0.9.0.1 --no-deps

# Install vLLM dependencies using constraints
RUN pip install -c /opt/constraints.txt \
    cmake \
    ninja \
    psutil \
    ray \
    sentencepiece \
    numpy \
    transformers \
    tokenizers \
    pillow \
    xformers \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing-extensions \
    filelock \
    partial-json-parser

# Clone SGLang at the specific commit (2nd occurrence of SHA)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 9d5fa68b903d295d2b39201d54905c6801f60f7f

# Verify we have the correct commit (3rd occurrence of SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="9d5fa68b903d295d2b39201d54905c6801f60f7f" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Git checkout failed. Expected $EXPECTED but got $ACTUAL"; \
        exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified SGLang commit: $ACTUAL"

# Install flashinfer from prebuilt wheels
RUN pip install flashinfer_python==0.2.5 -f https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.6.post1

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang && \
    pip install -e python --no-deps

# Install SGLang dependencies using constraints
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
    pyzmq \
    soundfile==0.13.1 \
    scipy \
    torchao==0.9.0 \
    transformers==4.52.3 \
    uvicorn \
    uvloop \
    xgrammar==0.1.19 \
    cuda-python \
    outlines \
    einops

# For MiniCPM models
RUN pip install datamodel_code_generator

# Install outlines-core
RUN pip install outlines-core==0.1.26

# Set environment variables for runtime
ENV PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH
ENV DEBIAN_FRONTEND=interactive

# Final verification
RUN python3 -c "import sglang; print('SGLang import successful')" && \
    python3 -c "import vllm; print('vLLM import successful')" && \
    python3 -c "import flashinfer; print('FlashInfer import successful')" && \
    python3 -c "import sgl_kernel; print('SGL Kernel import successful')" && \
    python3 -c "import outlines; print('Outlines import successful')"

# Sanity check: Import the main SGLang server module
RUN python3 -c "from sglang.srt.server import launch_engine; print('SGLang server module import successful')"

# Print versions for debugging
RUN echo "=== Installed Versions ===" && \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" && \
    python3 -c "import flashinfer; print(f'FlashInfer: {flashinfer.__version__}')" && \
    python3 -c "import outlines; print(f'Outlines: {outlines.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')" && \
    echo "SGLang commit: $(cat /opt/sglang_commit.txt)"

WORKDIR /sgl-workspace

# Default command
CMD ["/bin/bash"]