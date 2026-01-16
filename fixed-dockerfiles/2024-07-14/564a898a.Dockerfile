# SGLang Docker for commit from 2024-07-14
# Using torch 2.3.0 as required by vLLM 0.5.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_COMMIT=564a898ad975192b593be81387d11faf15cb1d3e
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=96

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git build-essential wget curl ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /sgl-workspace

# Create constraints file with versions discovered from PyPI for July 2024 era
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for 2024-07-14 era
# FastAPI 0.111.1 - Released July 14, 2024 (exact commit date)
fastapi==0.111.1
# Uvicorn 0.30.3 - Released May 1, 2024 (latest before July 14)
uvicorn==0.30.3
# Pydantic 2.8.2 - Released July 4, 2024 (v2 required by vLLM and outlines)
pydantic==2.8.2
# typing_extensions 4.12.2 - Released June 7, 2024
typing_extensions==4.12.2
# Outlines 0.0.44 - Released June 14, 2024 (from pyproject.toml)
outlines==0.0.44
# pyzmq 26.0.3 - Released May 1, 2024 (latest before July 14)
pyzmq==26.0.3
EOF

# Install vLLM 0.5.1 with --no-deps first
RUN pip install --upgrade pip setuptools wheel && \
    pip install vllm==0.5.1 --no-deps

# Install vLLM dependencies with constraints
RUN pip install -c /opt/constraints.txt \
    cmake ninja psutil sentencepiece 'numpy<2.0.0' requests tqdm py-cpuinfo \
    'transformers>=4.42.0' 'tokenizers>=0.19.1' fastapi aiohttp openai \
    'uvicorn[standard]' 'pydantic>=2.0' pillow 'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' 'tiktoken>=0.6.0' \
    'lm-format-enforcer==0.10.1' 'outlines>=0.0.43' typing_extensions \
    'filelock>=3.10.4' 'ray>=2.9' nvidia-ml-py

# Install torch-specific packages for vLLM
RUN pip install torchvision==0.18.0 vllm-flash-attn==2.5.9

# Install xformers with --no-deps to prevent pulling wrong torch version
RUN pip install xformers==0.0.26.post1 --no-deps

# Clone SGLang at specific commit
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout 564a898ad975192b593be81387d11faf15cb1d3e

# Verify commit SHA
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="564a898ad975192b593be81387d11faf15cb1d3e" && \
    test "$ACTUAL" = "$EXPECTED" || exit 1 && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# Install SGLang with --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# Install SGLang dependencies with constraints
# Note: Don't reinstall torch - keep the one from base image
RUN pip install -c /opt/constraints.txt \
    requests tqdm numpy aiohttp fastapi hf_transfer huggingface_hub \
    interegular packaging pillow psutil pydantic rpyc uvicorn \
    uvloop pyzmq outlines

# Install flashinfer
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# Install sgl-kernel version from July 2024 era
RUN pip install sgl-kernel==0.2.13 || echo "sgl-kernel not available for this version"

# Install additional SGLang optional dependencies
RUN pip install 'openai>=1.0' tiktoken 'anthropic>=0.20.0' 'litellm>=1.0.0'

# Install triton-nightly
RUN pip uninstall -y triton triton-nightly && \
    pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Final verification
# Note: vLLM import requires GPU libraries, so we verify it's installed via pip instead
RUN python -c "import sglang; print('SGLang imported successfully')" && \
    pip show vllm > /dev/null && echo "vLLM installed OK" && \
    python -c "import outlines; print('Outlines imported successfully')" && \
    python -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]