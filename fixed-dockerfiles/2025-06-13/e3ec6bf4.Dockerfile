# SGLang Docker Image
# Date: 2025-06-13
# Using torch 2.3.1 (June 2024 era) since torch 2.7.1 from pyproject.toml doesn't exist yet

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=e3ec6bf4b65a50e26e936a96adc7acc618292002
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Create constraints file with pinned versions discovered from PyPI
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for June 2024 era
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
pydantic-core==2.18.4
typing_extensions==4.12.2
outlines==0.0.44
pyzmq==26.0.3
msgspec==0.18.6
orjson==3.10.3
prometheus-client==0.20.0
hf_transfer==0.1.6
huggingface_hub==0.23.3
packaging==24.0
psutil==5.9.8
numpy==1.26.4
aiohttp==3.9.5
requests==2.32.3
tqdm==4.66.4
setproctitle==1.3.3
pillow==10.3.0
transformers==4.52.3
interegular==0.3.3
llguidance==0.7.11
partial_json_parser==0.2.1.1.post4
torchao==0.9.0
xgrammar==0.1.19
blobfile==3.0.0
compressed-tensors==0.4.0
datasets==2.19.2
modelscope==1.14.0
python-multipart==0.0.9
pynvml==11.5.0
soundfile==0.13.1
scipy==1.13.1
uvloop==0.19.0
einops==0.8.0
jinja2==3.1.4
lark==1.1.9
nest-asyncio==1.6.0
cloudpickle==3.0.0
diskcache==5.6.3
numba==0.60.0
referencing==0.35.1
jsonschema==4.22.0
pycountry==23.12.11
pyairports==2.1.1
EOF

WORKDIR /sgl-workspace

# Clone SGLang at the specific commit (1st hardcoded SHA)
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout e3ec6bf4b65a50e26e936a96adc7acc618292002

# Verify commit SHA (2nd hardcoded SHA)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e3ec6bf4b65a50e26e936a96adc7acc618292002" && \
    if [ "$ACTUAL" != "$EXPECTED" ]; then \
        echo "ERROR: Commit mismatch! Expected $EXPECTED but got $ACTUAL" && exit 1; \
    fi && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified commit: $ACTUAL"

# Install vLLM 0.4.2 (matching SGLang requirements for this era) with --no-deps
RUN pip install vllm==0.4.2 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    ninja \
    psutil \
    numpy \
    torch \
    ray \
    sentencepiece \
    transformers \
    tokenizers \
    filelock \
    pyzmq \
    uvloop \
    pydantic \
    pillow \
    prometheus-client \
    py-cpuinfo \
    pynvml \
    triton \
    cupy-cuda12x \
    nvidia-ml-py \
    peft

# Install sgl-kernel from PyPI
RUN pip install sgl-kernel==0.1.8.post1

# Build flashinfer from source since no prebuilt wheels for torch 2.3
RUN cd /tmp && \
    git clone https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    pip install . && \
    cd / && \
    rm -rf /tmp/flashinfer

# Install SGLang with --no-deps (3rd hardcoded SHA in path)
RUN cd /sgl-workspace/sglang/python && \
    pip install -e . --no-deps

# Install SGLang dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    aiohttp \
    requests \
    tqdm \
    numpy \
    IPython \
    setproctitle \
    blobfile \
    compressed-tensors \
    datasets \
    fastapi \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance \
    modelscope \
    msgspec \
    ninja \
    orjson \
    packaging \
    partial_json_parser \
    pillow \
    prometheus-client \
    psutil \
    pydantic \
    pynvml \
    python-multipart \
    pyzmq \
    soundfile \
    scipy \
    torchao \
    transformers \
    uvicorn \
    uvloop \
    xgrammar \
    einops \
    cuda-python

# Install datamodel_code_generator for MiniCPM models
RUN pip install datamodel_code_generator

# Final verification
RUN python3 -c "import sglang; print('SGLang imported successfully')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import flashinfer; print('flashinfer imported successfully')" && \
    python3 -c "import outlines; print('outlines imported successfully')" && \
    cat /opt/sglang_commit.txt && \
    echo "All imports successful!"

ENV DEBIAN_FRONTEND=interactive

ENTRYPOINT ["python", "-m", "sglang.launch_server"]