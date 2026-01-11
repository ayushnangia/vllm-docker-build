# Dockerfile for SGLang commit ddcf9fe3beacd8aed573c711942194dd02350da4 (2025-02-20)
# torch 2.5.1 with CUDA 12.1 for vLLM 0.7.2

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96
# First occurrence of SHA: Environment variable
ENV SGLANG_COMMIT=ddcf9fe3beacd8aed573c711942194dd02350da4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget curl git vim build-essential cmake ninja-build \
    libibverbs-dev rdma-core infiniband-diags openssh-server \
    perftest ibverbs-providers libibumad3 libibverbs1 \
    libnl-3-200 libnl-route-3-200 librdmacm1 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install torch 2.5.1 with CUDA 12.1
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install xformers for torch 2.5.1
RUN pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with discovered versions from PyPI for 2025-02-20
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2025-02-20 era
fastapi==0.115.8
uvicorn==0.34.0
pydantic==2.10.6
typing_extensions==4.12.2
outlines==0.1.11
pyzmq==26.2.1
uvloop==0.21.0
aiohttp==3.11.12
orjson==3.10.15
transformers==4.48.3
xgrammar==0.1.10
torchao==0.8.0
prometheus-client==0.21.1
huggingface-hub==0.29.1
lm-format-enforcer==0.10.10
lark==1.2.2
EOF

# Install vLLM 0.7.2 with --no-deps
RUN pip install vllm==0.7.2 --no-deps

# Install vLLM dependencies from requirements-common.txt and requirements-cuda.txt
RUN pip install -c /opt/constraints.txt \
    psutil sentencepiece 'numpy<2.0.0' 'requests>=2.26.0' tqdm blake3 py-cpuinfo \
    'tokenizers>=0.19.1' protobuf 'fastapi>=0.107.0,!=0.113.*,!=0.114.0' aiohttp \
    'openai>=1.52.0' 'uvicorn[standard]' 'pydantic>=2.9' 'prometheus_client>=0.18.0' \
    pillow 'prometheus-fastapi-instrumentator>=7.0.0' 'tiktoken>=0.6.0' \
    'lm-format-enforcer>=0.10.9,<0.11' 'outlines==0.1.11' 'lark==1.2.2' \
    'xgrammar>=0.1.6' 'typing_extensions>=4.10' 'filelock>=3.16.1' \
    partial-json-parser pyzmq msgspec 'gguf==0.10.0' importlib_metadata \
    'mistral_common[opencv]>=1.5.0' pyyaml 'six>=1.16.0' 'setuptools>=74.1.1' \
    einops 'compressed-tensors==0.9.1' 'depyf==0.18.0' cloudpickle \
    'nvidia-ml-py>=12.560.30' 'ray[default]>=2.9'

# Apply constraints to ensure correct versions
RUN pip install -c /opt/constraints.txt \
    fastapi uvicorn pydantic typing_extensions outlines pyzmq uvloop aiohttp orjson \
    transformers xgrammar torchao prometheus-client huggingface-hub lm-format-enforcer lark

# Install flashinfer from wheels
RUN pip install flashinfer==0.2.5 --find-links https://flashinfer.ai/whl/cu121/torch2.5/flashinfer-python/

# Second occurrence of SHA: Clone SGLang at specific commit
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout ddcf9fe3beacd8aed573c711942194dd02350da4

# Third occurrence of SHA: Verify correct commit and write to file
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ddcf9fe3beacd8aed573c711942194dd02350da4" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH: expected $EXPECTED but got $ACTUAL" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Build and install sgl-kernel from source (not available on PyPI until April 2025)
# sgl-kernel is part of SGLang, built when installing the package
RUN cd /sgl-workspace/sglang/python && \
    pip install ninja && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 python3 setup.py build_ext --inplace

# Install SGLang with --no-deps to avoid version conflicts
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# Install remaining SGLang dependencies from pyproject.toml using constraints
RUN pip install -c /opt/constraints.txt \
    requests tqdm numpy IPython setproctitle \
    decord hf_transfer huggingface_hub interegular modelscope \
    packaging python-multipart torchao xgrammar ninja \
    transformers cuda-python

# Install optional dependencies for OpenAI/Anthropic/LiteLLM support
RUN pip install -c /opt/constraints.txt \
    'openai>=1.0' tiktoken 'anthropic>=0.20.0' 'litellm>=1.0.0'

# Final verification
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import vllm; print('vLLM imported successfully')" && \
    python3 -c "import outlines; print('Outlines imported successfully')" && \
    python3 -c "import flashinfer; print('FlashInfer imported successfully')" && \
    python3 -c "import sglang; print('SGLang imported successfully')"

# Set environment to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

WORKDIR /workspace

# Entrypoint
CMD ["/bin/bash"]