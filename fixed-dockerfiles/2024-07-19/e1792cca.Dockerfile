# Base image for torch 2.4.x with CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl wget sudo \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (Ubuntu 20.04 deadsnakes PPA is broken)
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

# Verify Python installation
RUN python3 --version && python3 -m pip --version

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# CUDA compatibility workaround
RUN ldconfig /usr/local/cuda-12.1/compat/

# Pre-install torch 2.4.0 with CUDA 12.1
RUN pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vllm 0.5.1 (as specified in pyproject.toml)
RUN pip3 install vllm==0.5.1

# HARDCODE the commit SHA (1st occurrence)
ENV SGLANG_COMMIT=e1792cca2491af86f29782a3b83533a6566ac75b

# Clone SGLang and checkout EXACT commit (2nd occurrence - hardcoded)
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout e1792cca2491af86f29782a3b83533a6566ac75b

# VERIFY commit - compare against HARDCODED value (3rd occurrence)
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="e1792cca2491af86f29782a3b83533a6566ac75b" && \
    echo "Expected: $EXPECTED" && \
    echo "Actual:   $ACTUAL" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "FATAL: COMMIT MISMATCH!" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt && \
    echo "Verified: SGLang at commit $ACTUAL"

# Install SGLang from source
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

# Replace triton with triton-nightly (common for this era)
RUN pip3 uninstall -y triton triton-nightly || true \
    && pip3 install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Final verification
RUN python3 -c "import sglang; print('SGLang import OK')" \
    && python3 -c "import vllm; print('vLLM import OK')"

# Set working directory
WORKDIR /sgl-workspace

ENV DEBIAN_FRONTEND=interactive

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]