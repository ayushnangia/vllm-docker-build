FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set build arguments and environment
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set commit SHA - 1st occurrence (hardcoded)
ENV SGLANG_COMMIT=c98e84c21e4313d7d307425ca43e61753a53a9f7

# Install system dependencies and Python 3.10
RUN apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        git \
        curl \
        wget \
        build-essential \
        libibverbs-dev \
        libssl-dev \
        libffi-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        liblzma-dev \
        zlib1g-dev \
        cmake \
        ninja-build && \
    cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    cd / && \
    rm -rf /tmp/Python-3.10.14* && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify Python installation
RUN python3 --version && python3 -m pip --version

# Set working directory
WORKDIR /sgl-workspace

# Install PyTorch 2.4.0 with CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Create constraints file with versions from October 2024 era
# These versions were discovered via PyPI searches for October 6, 2024
RUN cat > /opt/constraints.txt <<'EOF'
# Core dependencies for October 2024
# FastAPI 0.114.0 (Sept 6, 2024) - supports both pydantic v1 and v2
fastapi==0.114.0
uvicorn==0.30.6
# Pydantic 2.9.2 (Sept 17, 2024) - required by vLLM >= 2.8 and outlines >= 2.0
pydantic==2.9.2
pydantic-core==2.23.4
# typing_extensions 4.12.2 (June 7, 2024) - avoids Sentinel issue in 4.14+
typing_extensions==4.12.2
# Outlines 0.0.46 (June 22, 2024) - requires pydantic >= 2.0
outlines==0.0.46
pyzmq==26.2.0
# vLLM dependencies
numpy==1.26.4
requests==2.32.3
tqdm==4.66.5
transformers==4.44.2
tokenizers==0.19.1
protobuf==5.28.0
aiohttp==3.10.5
openai==1.46.0
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
psutil==6.0.0
sentencepiece==0.2.0
py-cpuinfo==9.0.0
huggingface_hub==0.24.6
packaging==24.1
interegular==0.3.3
hf_transfer==0.1.8
decord==0.6.0
python-multipart==0.0.9
uvloop==0.20.0
torchao==0.5.0
modelscope==1.18.0
ray==2.35.0
nvidia-ml-py==12.560.30
EOF

# Install vLLM 0.5.5 without dependencies
RUN pip3 install --no-cache-dir vllm==0.5.5 --no-deps

# Install vLLM dependencies with constraints
RUN pip3 install --no-cache-dir -c /opt/constraints.txt \
    psutil sentencepiece 'numpy<2.0.0' requests tqdm py-cpuinfo \
    'transformers>=4.43.2' 'tokenizers>=0.19.1' protobuf \
    fastapi aiohttp 'openai>=1.0' 'uvicorn[standard]' \
    'pydantic>=2.8' pillow 'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' 'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.6' 'outlines>=0.0.43,<0.1' \
    'typing_extensions>=4.10' 'filelock>=3.10.4' pyzmq msgspec \
    librosa soundfile 'gguf==0.9.1' importlib_metadata

# Install additional vLLM CUDA dependencies
RUN pip3 install --no-cache-dir \
    'ray>=2.9' nvidia-ml-py \
    'xformers==0.0.27.post2' \
    'vllm-flash-attn==2.6.1'

# Clone SGLang repository at specific commit - 2nd occurrence (hardcoded)
RUN git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git checkout c98e84c21e4313d7d307425ca43e61753a53a9f7

# Verify correct commit and save commit SHA - 3rd occurrence (hardcoded)
RUN cd /sgl-workspace/sglang && \
    ACTUAL_COMMIT=$(git rev-parse HEAD) && \
    EXPECTED_COMMIT="c98e84c21e4313d7d307425ca43e61753a53a9f7" && \
    echo "Expected: $EXPECTED_COMMIT" && \
    echo "Actual:   $ACTUAL_COMMIT" && \
    if [ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]; then \
        echo "FATAL: COMMIT MISMATCH! Expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT" && \
        exit 1; \
    fi && \
    echo "$ACTUAL_COMMIT" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL_COMMIT"

# Set build environment for H100
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install flashinfer from wheel repository for torch2.4 cu121
RUN pip3 install --no-cache-dir flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ || \
    echo "flashinfer wheel not found, will be built from source if needed"

# Patch pyproject.toml to remove already-installed dependencies
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torch[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"torchao[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml

# Install SGLang with dependencies using constraints
RUN cd /sgl-workspace/sglang/python && \
    pip3 install --no-cache-dir -e . --no-deps && \
    pip3 install --no-cache-dir -c /opt/constraints.txt \
        requests tqdm numpy \
        aiohttp decord fastapi hf_transfer huggingface_hub interegular \
        packaging pillow psutil pydantic python-multipart \
        uvicorn uvloop pyzmq \
        'outlines>=0.0.44' modelscope

# Install additional optional dependencies
RUN pip3 install --no-cache-dir \
    'openai>=1.0' tiktoken \
    'anthropic>=0.20.0' \
    'litellm>=1.0.0' \
    datamodel_code_generator

# Clean up pip cache
RUN pip3 cache purge

# Final verification - ensure all critical imports work
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')" && \
    python3 -c "import xformers; print('xformers import OK')" && \
    python3 -c "import outlines; print('outlines import OK')" && \
    python3 -c "import pydantic; print(f'pydantic: {pydantic.__version__}')"

# Set interactive mode back
ENV DEBIAN_FRONTEND=interactive

# Add a note about the build
RUN echo "SGLang commit: ${SGLANG_COMMIT}" >> /opt/sglang_info.txt && \
    echo "Build date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> /opt/sglang_info.txt && \
    echo "PyTorch: 2.4.0" >> /opt/sglang_info.txt && \
    echo "vLLM: 0.5.5" >> /opt/sglang_info.txt && \
    echo "Pydantic: 2.9.2" >> /opt/sglang_info.txt && \
    echo "Outlines: 0.0.46" >> /opt/sglang_info.txt

# Set the working directory
WORKDIR /sgl-workspace

# Verification command
CMD ["python3", "-c", "import sglang; print('SGLang installation successful')"]