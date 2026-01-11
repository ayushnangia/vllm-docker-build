# Base image for torch 2.4.x
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
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
RUN pip3 install --upgrade pip setuptools wheel

# Install cmake and ninja for building packages
RUN apt-get update && apt-get install -y cmake ninja-build && rm -rf /var/lib/apt/lists/*

# Pre-install torch 2.4.0 with CUDA 12.1
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.5.1 as specified in pyproject.toml
RUN pip3 install vllm==0.5.1

# Build flashinfer from source (for torch 2.4.0)
RUN pip3 install ninja numpy packaging \
    && git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer \
    && cd /tmp/flashinfer \
    && git checkout v0.1.5 \
    && cd python \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip3 install --no-build-isolation . \
    && rm -rf /tmp/flashinfer

# HARDCODE the commit SHA (occurrence 1/3)
ENV SGLANG_COMMIT=ac971ff633de330de3ded7f7475caaf7cd5bbdcd

# Clone SGLang and checkout EXACT commit (occurrence 2/3)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout ac971ff633de330de3ded7f7475caaf7cd5bbdcd

# VERIFY commit - compare against HARDCODED value (occurrence 3/3)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="ac971ff633de330de3ded7f7475caaf7cd5bbdcd" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Patch pyproject.toml to remove already-installed deps
RUN cd /sgl-workspace/sglang && \
    if [ -f python/pyproject.toml ]; then \
        sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"flashinfer_python[^"]*",*//g' python/pyproject.toml && \
        sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml; \
    fi

# Install SGLang from source
WORKDIR /sgl-workspace/sglang
RUN cd python && pip3 install -e ".[all]"

# Install Triton nightly (often needed for compatibility)
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Verify installation
RUN python3 -c "import sglang; print('SGLang import OK')"
RUN python3 -c "import flashinfer; print('Flashinfer import OK')"
RUN python3 -c "import vllm; print('vLLM import OK')"

# Set working directory
WORKDIR /sgl-workspace/sglang

# Entry point
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]