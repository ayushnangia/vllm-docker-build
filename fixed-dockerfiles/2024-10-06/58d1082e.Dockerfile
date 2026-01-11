# Fixed Dockerfile for SGLang commit 58d1082e392cabbf26c404cb7ec18e4cb51b99e9
# Date: 2024-10-06
# vLLM 0.5.5 requires torch 2.4.0, CUDA 12.1

# Use PyTorch base image for torch 2.4.0
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Build settings
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# HARDCODE commit SHA (1st occurrence)
ENV SGLANG_COMMIT=58d1082e392cabbf26c404cb7ec18e4cb51b99e9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    ninja-build \
    libibverbs-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Create constraints file with discovered versions from PyPI for 2024-10-06 era
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies - versions discovered from PyPI for 2024-10-06 era
fastapi==0.114.2
uvicorn==0.30.6
pydantic==2.9.2
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.2.0

# vLLM dependencies from requirements-common.txt
psutil==6.0.0
sentencepiece==0.2.0
numpy==1.26.4
requests==2.32.3
tqdm==4.66.5
py-cpuinfo==9.0.0
transformers==4.44.2
tokenizers==0.19.1
protobuf==5.28.0
aiohttp==3.10.5
openai==1.45.0
pillow==10.4.0
prometheus_client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
tiktoken==0.7.0
lm-format-enforcer==0.10.6
filelock==3.16.0
msgspec==0.18.6
librosa==0.10.2.post1
soundfile==0.12.1
gguf==0.9.1
importlib_metadata==8.4.0

# vLLM CUDA dependencies from requirements-cuda.txt
ray==2.36.0
nvidia-ml-py==12.560.30
torchvision==0.19.0
xformers==0.0.27.post2
vllm-flash-attn==2.6.1

# SGLang specific dependencies
decord==0.6.0
hf_transfer==0.1.8
huggingface_hub==0.24.6
interegular==0.3.3
packaging==24.1
python-multipart==0.0.9
torchao==0.5.0
uvloop==0.20.0
modelscope==1.18.0
EOF

# Verify torch is installed (from base image)
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Install vLLM with --no-deps to control dependency versions
RUN pip install vllm==0.5.5 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
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
    lm-format-enforcer \
    "outlines>=0.0.43,<0.1" \
    "typing_extensions>=4.10" \
    "filelock>=3.10.4" \
    pyzmq \
    msgspec \
    librosa \
    soundfile \
    gguf \
    importlib_metadata \
    "ray>=2.9" \
    nvidia-ml-py \
    torchvision \
    xformers \
    vllm-flash-attn

# Install flashinfer from official wheels (version 0.1.6 was available in Oct 2024)
RUN pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/

# Clone SGLang at specific commit (2nd occurrence of SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout 58d1082e392cabbf26c404cb7ec18e4cb51b99e9

# Verify commit SHA and save to file (3rd occurrence of SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="58d1082e392cabbf26c404cb7ec18e4cb51b99e9" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "Error: Expected commit $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang with --no-deps
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    requests \
    tqdm \
    numpy \
    aiohttp \
    decord \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    packaging \
    pillow \
    psutil \
    pydantic \
    python-multipart \
    torchao \
    uvicorn \
    uvloop \
    pyzmq \
    modelscope

# Install zmq package (different from pyzmq)
RUN pip install zmq

# For openbmb/MiniCPM models
RUN pip install datamodel_code_generator

# Sanity check: verify imports work
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" && \
    python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import outlines; print(f'Outlines: {outlines.__version__}')" && \
    python3 -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')" && \
    python3 -c "import flashinfer; print('Flashinfer imported successfully')"

# Clear pip cache
RUN python3 -m pip cache purge

# Set working directory
WORKDIR /workspace

# Label with commit info
LABEL org.opencontainers.image.revision="58d1082e392cabbf26c404cb7ec18e4cb51b99e9"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.description="SGLang Docker image for commit 58d1082e392cabbf26c404cb7ec18e4cb51b99e9"

ENV DEBIAN_FRONTEND=interactive

# Set default command
CMD ["/bin/bash"]