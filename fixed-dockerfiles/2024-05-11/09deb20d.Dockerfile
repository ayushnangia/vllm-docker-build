# SGLang @ 09deb20d (2024-05-11 era)
# Time-frozen deps: CUDA 12.1 + torch 2.3.0 + vLLM 0.4.2 + outlines 0.0.34 + pydantic 2.7.1

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Ensure CUDA wheels index for torch ecosystem packages
RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0
RUN pip install --index-url https://download.pytorch.org/whl/cu121 xformers==0.0.26.post1

# ---- Pin the "May 2024" userland deps so pip doesn't pull 2026 versions ----
RUN cat > /opt/constraints-2024-05.txt <<'EOF'
fastapi==0.111.0
uvicorn==0.29.0

# Pydantic v2 era (May 2024-ish)
pydantic==2.7.1
typing_extensions==4.11.0

# vLLM 0.4.2 requires outlines==0.0.34 (not 0.0.39!)
outlines==0.0.34

# misc
pyzmq==26.0.3
EOF

# ---- flashinfer (SGLang wants flashinfer>=0.0.4; build from source for Hopper) ----
RUN pip install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.1.2 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# ---- vLLM 0.4.2 (May 5, 2024) ----
# Install vLLM wheel without letting it drag modern deps; we install its Python deps pinned.
RUN pip install vllm==0.4.2 --no-deps

# vLLM runtime deps (from vLLM 0.4.2 requirements)
RUN pip install -c /opt/constraints-2024-05.txt \
    numpy requests psutil sentencepiece py-cpuinfo filelock packaging \
    "transformers==4.40.2" "tokenizers==0.19.1" \
    "uvicorn[standard]==0.29.0" fastapi==0.111.0 pydantic==2.7.1 \
    prometheus_client prometheus-fastapi-instrumentator>=7.0.0 \
    "ray>=2.9" nvidia-ml-py openai tiktoken==0.6.0 \
    lm-format-enforcer==0.9.8 cmake>=3.21 \
    outlines==0.0.34

# ---- SGLang @ exact commit (HARDCODED in 3 places) ----
# 1st occurrence: ENV
ENV SGLANG_COMMIT=09deb20deef8181a23f66c933ea74b86fee47366

WORKDIR /sgl-workspace
# 2nd occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout 09deb20deef8181a23f66c933ea74b86fee47366

# 3rd occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="09deb20deef8181a23f66c933ea74b86fee47366" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang itself without deps (we control deps explicitly)
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# SGLang srt-ish deps (matching pyproject extras, but pinned)
# outlines already installed with vLLM deps (0.0.34 as vLLM requires)
RUN pip install -c /opt/constraints-2024-05.txt \
    aiohttp rpyc uvloop interegular pillow packaging \
    pyzmq

# Sanity check - use importlib.metadata for versions (some packages lack __version__)
RUN python -c "import torch; print('torch', torch.__version__)" && \
    python -c "import importlib.metadata as m; print('pydantic', m.version('pydantic')); print('typing_extensions', m.version('typing_extensions')); print('outlines', m.version('outlines')); print('fastapi', m.version('fastapi'))" && \
    python -c "import vllm; print('vllm import OK')" && \
    python -c "import flashinfer; print('flashinfer import OK')" && \
    python -c "import sglang; print('sglang import OK')"

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
