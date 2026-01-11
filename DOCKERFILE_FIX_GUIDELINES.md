# SGLang Dockerfile Fix Guidelines

## Core Principle: DISCOVER and TIME-FREEZE

Building old commits requires **discovering** the correct package versions from that era, then **time-freezing** them with a constraints file. Never hardcode - always discover by checking PyPI and repos.

---

## THE TIME-FREEZE PATTERN

### Why This Matters

Modern pip installs latest versions of unbounded deps, breaking old code. You must DISCOVER what versions existed at the commit date.

### The Solution: Constraints File + --no-deps

```dockerfile
# 1. Create constraints file with versions YOU DISCOVERED
RUN cat > /opt/constraints.txt <<'EOF'
# Each version below was DISCOVERED by checking PyPI release history
# for the version that was latest ON OR BEFORE the commit date
fastapi==<DISCOVERED_VERSION>
pydantic==<DISCOVERED_VERSION>
outlines==<DISCOVERED_VERSION>
typing_extensions==<DISCOVERED_VERSION>
uvicorn==<DISCOVERED_VERSION>
EOF

# 2. Install main packages with --no-deps
RUN pip install vllm==<VERSION_FROM_PYPROJECT> --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# 3. Install ALL deps with constraints enforced
RUN pip install -c /opt/constraints.txt <ALL_DISCOVERED_DEPS>
```

---

## MANDATORY: How to Discover Versions (DO NOT SKIP)

### Step 1: Check PyPI Release History for EACH Package

For every problematic package (fastapi, pydantic, outlines, typing_extensions, uvicorn):

```
WebFetch https://pypi.org/project/fastapi/#history
WebFetch https://pypi.org/project/pydantic/#history
WebFetch https://pypi.org/project/outlines/#history
WebFetch https://pypi.org/project/typing-extensions/#history
```

**Find the version that was released ON OR BEFORE the commit date.**

### Step 2: Check for Breaking Changes

WebSearch for each package to understand version boundaries:
```
WebSearch "<package_name> changelog breaking changes"
WebSearch "<package_name> version compatibility"
WebSearch "fastapi pydantic version requirement"
```

### Step 3: Verify Compatibility Between Packages

The versions you discovered must work together:
```
WebSearch "fastapi <version> pydantic <version> compatible"
```

### Step 4: Clone and Read Actual Requirements

```bash
# Setup unique temp dir
WORK_DIR="/tmp/explore-${SHORT_HASH}"
mkdir -p "$WORK_DIR" && cd "$WORK_DIR"

# Clone SGLang at commit
git clone https://github.com/sgl-project/sglang.git
cd sglang && git checkout <COMMIT_SHA>
cat python/pyproject.toml

# Clone vLLM at required version
cd "$WORK_DIR"
git clone --depth 1 --branch v<VERSION> https://github.com/vllm-project/vllm.git
cat vllm/requirements*.txt
```

---

## VERIFY THREE TIMES (MANDATORY)

### Before Writing Dockerfile:
1. **First check:** WebFetch PyPI history for each package, note exact versions and release dates
2. **Second check:** WebSearch for breaking changes between versions you found
3. **Third check:** Verify the versions work together (check compatibility)

### After Writing Dockerfile:
1. **First verify:** Read back the file, confirm all versions match your discoveries
2. **Second verify:** Count SHA occurrences (must be EXACTLY 3)
3. **Third verify:** Trace each pinned version back to your PyPI/WebSearch discovery

---

## Dockerfile Structure

```dockerfile
FROM <BASE_IMAGE_BASED_ON_TORCH_VERSION>

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Torch (use CUDA index, version from pyproject.toml)
RUN pip install --index-url https://download.pytorch.org/whl/cu<CUDA> torch==<DISCOVERED>
RUN pip install --index-url https://download.pytorch.org/whl/cu<CUDA> xformers==<DISCOVERED>

# TIME-FREEZE constraints (versions YOU discovered from PyPI)
RUN cat > /opt/constraints.txt <<'EOF'
fastapi==<DISCOVERED_FROM_PYPI>
pydantic==<DISCOVERED_FROM_PYPI>
outlines==<DISCOVERED_FROM_PYPI>
typing_extensions==<DISCOVERED_FROM_PYPI>
uvicorn==<DISCOVERED_FROM_PYPI>
EOF

# Flashinfer (check flashinfer.ai/whl/ for wheels, else build from source)
RUN pip install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && git checkout <DISCOVERED_TAG> && \
    cd python && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# vLLM with --no-deps (version from pyproject.toml)
RUN pip install vllm==<VERSION_FROM_PYPROJECT> --no-deps

# vLLM deps (discovered from vllm/requirements*.txt, with constraints)
RUN pip install -c /opt/constraints.txt \
    <DEPS_FROM_VLLM_REQUIREMENTS>

# SGLang @ exact commit (HARDCODE SHA in 3 places)
ENV SGLANG_COMMIT=<FULL_40_CHAR_SHA>

WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout <FULL_40_CHAR_SHA>

RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="<FULL_40_CHAR_SHA>" && \
    test "$ACTUAL" = "$EXPECTED" || exit 1 && \
    echo "$ACTUAL" > /opt/sglang_commit.txt

# SGLang with --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# SGLang deps (discovered from pyproject.toml, with constraints)
RUN pip install -c /opt/constraints.txt \
    <DEPS_FROM_SGLANG_PYPROJECT>

# Verify all imports
RUN python -c "import torch, vllm, sglang; print('All imports OK')"

WORKDIR /sgl-workspace/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
```

---

## Base Image Selection

Discover torch version from pyproject.toml, then:
- torch 2.1.x-2.3.x → `pytorch/pytorch:<version>-cuda12.1-cudnn8-devel`
- torch 2.4.x-2.5.x → `nvidia/cuda:12.1.1-devel-ubuntu20.04` (install Python manually)
- torch 2.6.x+ → `nvidia/cuda:12.4.1-devel-ubuntu22.04` (install Python manually)

---

## Flashinfer Handling

### Check wheel availability:
```
WebFetch https://flashinfer.ai/whl/cu<CUDA>/torch<MAJOR>.<MINOR>/
```

If wheel exists → `pip install flashinfer -i <URL>`
If no wheel → build from source with discovered compatible tag

---

## sgl-kernel Handling

### Check PyPI:
```
WebFetch https://pypi.org/simple/sgl-kernel/
```

If version exists → `pip install sgl-kernel==<VERSION>`
If not → build from source

---

## Common Discovery Patterns

### For pydantic v1 vs v2:
```
WebSearch "fastapi <version> pydantic requirement"
WebFetch https://pypi.org/project/fastapi/<version>/
```
Check if fastapi version requires pydantic v1 or v2.

### For xformers:
```
WebFetch https://pypi.org/project/xformers/#history
```
Find xformers version released near commit date, verify torch compatibility.

### For outlines:
```
WebFetch https://pypi.org/project/outlines/#history
WebSearch "outlines <version> api changes"
```
Major API changes happened around 0.1.0.

---

## Build Settings

- **TORCH_CUDA_ARCH_LIST="9.0"** - H100 only (fastest build)
- **MAX_JOBS=96** - Use all 96 cores

---

## Output Location

Write fixed Dockerfile to:
```
fixed-dockerfiles/{pr_date}/{short_hash}.Dockerfile
```

---

## Final Checklist

Before finishing, verify:

- [ ] All versions discovered from PyPI/repos (not guessed)
- [ ] SHA appears exactly 3 times (ENV, checkout, verification)
- [ ] --no-deps used for vLLM and SGLang
- [ ] Constraints file created with discovered versions
- [ ] All deps installed with `-c /opt/constraints.txt`
- [ ] Imports verified at end of Dockerfile
