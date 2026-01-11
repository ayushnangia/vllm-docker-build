# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (deadsnakes PPA is broken on Ubuntu 20.04)
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar -xf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && ldconfig \
    && ln -sf /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 \
    && cd .. && rm -rf Python-3.10.14*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Pre-install torch 2.4.1 with CUDA 12.1
RUN pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer from wheel (available for torch 2.4 + CUDA 12.1)
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install vLLM 0.5.5 as specified in pyproject.toml
RUN pip3 install vllm==0.5.5

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9

# Clone SGLang and checkout EXACT commit (occurrence 2/3 - hardcoded SHA)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout 5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove vllm since we pre-installed it
RUN sed -i 's/"vllm[^"]*",*//g' /sgl-workspace/sglang/python/pyproject.toml && \
    sed -i 's/"torch",*//g' /sgl-workspace/sglang/python/pyproject.toml

# Install SGLang from source
WORKDIR /sgl-workspace/sglang
RUN cd python && pip3 install -e ".[all]"

# Install triton-nightly to avoid conflicts
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('Flashinfer import OK')" && \
    python3 -c "import vllm; print('vLLM import OK')"

# Clear pip cache
RUN pip3 cache purge

WORKDIR /sgl-workspace/sglang

# Set environment to interactive for runtime
ENV DEBIAN_FRONTEND=interactive

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]