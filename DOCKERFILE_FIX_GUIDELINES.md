# SGLang Dockerfile Fix Guidelines

## Core Principle: TIME-FREEZE Dependencies

Building old commits in 2026 requires **time-freezing** the Python ecosystem to the commit's era. Without this, pip installs modern versions that break old code.

**Example:** A May 2024 commit had unbounded `fastapi` and `outlines` deps. In 2026, pip pulls FastAPI 0.126+ (forces pydantic v2) and Outlines 1.x (different API), breaking everything.

---

## THE TIME-FREEZE PATTERN (Most Important Section)

### Why This Matters

| Package | May 2024 | 2026 | Breaking Change |
|---------|----------|------|-----------------|
| FastAPI | 0.111.0 | 0.128+ | Forces pydantic v2 (>=0.126) |
| Pydantic | 1.10.x | 2.x | API changes, typing_extensions.Sentinel |
| Outlines | 0.0.39 | 1.x | Complete API rewrite |
| typing_extensions | 4.11.0 | 4.14+ | Sentinel class added in 4.12 |

### The Solution: Constraints File + --no-deps

```dockerfile
# 1. Create constraints file with EXACT versions from commit's era
RUN cat > /opt/constraints-YYYY-MM.txt <<'EOF'
# Find these versions by checking PyPI release dates near commit date
fastapi==0.111.0
uvicorn==0.29.0
pydantic==1.10.13
outlines==0.0.39
typing_extensions==4.11.0
pyzmq==26.0.3
EOF

# 2. Install main packages with --no-deps (prevents transitive dep explosion)
RUN pip install vllm==0.4.2 --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# 3. Install ALL dependencies with constraints enforced
RUN pip install -c /opt/constraints-YYYY-MM.txt \
    numpy requests psutil transformers tokenizers \
    fastapi uvicorn pydantic \
    aiohttp rpyc uvloop outlines pyzmq
```

### How to Find Era-Appropriate Versions

1. **Check PyPI release dates:**
   - https://pypi.org/project/fastapi/#history
   - https://pypi.org/project/outlines/#history
   - Find versions released BEFORE or ON the commit date

2. **Key breakpoints to know:**
   - `fastapi >= 0.126.0` → forces pydantic v2 (dropped v1 support)
   - `outlines >= 0.1.0` → major API changes
   - `typing_extensions >= 4.12` → added Sentinel class
   - `pydantic >= 2.0` → requires pydantic-core, new API

3. **Safe defaults by era:**

   **Jan-Jul 2024 (pydantic v1 era):**
   ```
   fastapi<0.126.0
   pydantic>=1.10,<2.0
   typing_extensions>=4.5,<4.12
   outlines<0.1.0
   ```

   **Aug 2024+ (pydantic v2 transition):**
   ```
   fastapi>=0.100.0
   pydantic>=2.0
   typing_extensions>=4.12
   ```

### Complete Example: May 2024 Commit

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Torch ecosystem (use CUDA index)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0
RUN pip install --index-url https://download.pytorch.org/whl/cu121 xformers==0.0.26.post1

# TIME-FREEZE: Pin deps to May 2024 versions
RUN cat > /opt/constraints-2024-05.txt <<'EOF'
fastapi==0.111.0
uvicorn==0.29.0
pydantic==1.10.13
outlines==0.0.39
typing_extensions==4.11.0
pyzmq==26.0.3
EOF

# Build flashinfer from source (for H100)
RUN pip install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && git checkout v0.1.2 && \
    cd python && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    rm -rf /tmp/flashinfer

# vLLM with --no-deps
RUN pip install vllm==0.4.2 --no-deps

# vLLM deps (with constraints)
RUN pip install -c /opt/constraints-2024-05.txt \
    numpy requests psutil sentencepiece py-cpuinfo filelock packaging \
    transformers==4.40.2 tokenizers==0.19.1 \
    uvicorn[standard]==0.29.0 fastapi==0.111.0 pydantic==1.10.13

# SGLang @ exact commit
ENV SGLANG_COMMIT=<FULL_SHA>
WORKDIR /sgl-workspace
RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout <FULL_SHA>

# SGLang with --no-deps
RUN pip install -e /sgl-workspace/sglang/python --no-deps

# SGLang deps (with constraints)
RUN pip install -c /opt/constraints-2024-05.txt \
    aiohttp rpyc uvloop interegular pillow packaging pyzmq outlines==0.0.39

# Verify
RUN python -c "import torch, vllm, flashinfer, sglang; print('All imports OK')"
```

---

## DISCOVERY: Find What Versions to Pin

Before writing ANY Dockerfile, explore the actual repos to discover what's needed.

### Step 0: Explore the Repos (DO THIS FIRST)

Before writing ANY Dockerfile code, you MUST clone and explore the actual repositories to understand dependencies. Use a unique temp directory per commit for parallel safety.

### 0a. Setup Exploration Environment

```bash
# Create unique temp directory (use short hash for parallel safety)
SHORT_HASH="${COMMIT_SHA:0:8}"
WORK_DIR="/tmp/explore-${SHORT_HASH}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
```

### 0b. Clone and Explore SGLang at the Exact Commit

```bash
git clone https://github.com/sgl-project/sglang.git sglang
cd sglang
git checkout ${COMMIT_SHA}

# Find where pyproject.toml is located
find . -name "pyproject.toml" -o -name "setup.py" | head -10

# Read the actual pyproject.toml
cat python/pyproject.toml 2>/dev/null || cat pyproject.toml 2>/dev/null

# Note the EXACT versions required for:
# - torch (look in [project.optional-dependencies] srt section)
# - vllm
# - flashinfer or flashinfer_python
# - sgl-kernel
# - transformers
# - pydantic
```

### 0c. Explore vLLM Dependencies (CRITICAL)

If pyproject.toml specifies vLLM, you MUST check what vLLM actually requires:

```bash
cd "$WORK_DIR"

# Get the vLLM version from SGLang's pyproject.toml (e.g., vllm==0.4.2)
VLLM_VERSION="v0.4.2"  # Replace with actual version

# Clone vLLM at that exact version
git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git vllm

# Read vLLM's ACTUAL requirements
cat vllm/requirements-cuda.txt 2>/dev/null || cat vllm/requirements.txt
cat vllm/pyproject.toml 2>/dev/null | head -100

# Look for:
# - torch version requirement
# - xformers version requirement
# - pydantic version requirement
# - Any other critical deps
```

### 0d. Explore Flashinfer (if needed)

If flashinfer is required, check what versions are available:

```bash
cd "$WORK_DIR"

# Clone flashinfer to see available tags
git clone https://github.com/flashinfer-ai/flashinfer.git flashinfer
cd flashinfer
git tag | sort -V | tail -20

# Check a specific version's requirements
git checkout v0.1.2
cat python/pyproject.toml 2>/dev/null
cat setup.py 2>/dev/null | head -50
```

### 0e. Check xformers Compatibility

xformers is tightly coupled to torch. You MUST verify:

```bash
# WebSearch for the exact compatible version
# Search: "xformers torch 2.3.0 compatible version"
# Search: "xformers 0.0.26 torch requirement"

# Or clone xformers and check
cd "$WORK_DIR"
git clone --depth 1 https://github.com/facebookresearch/xformers.git xformers
cat xformers/requirements.txt
```

### 0f. Cleanup After Exploration

```bash
rm -rf "$WORK_DIR"
```

---

## Step 1: Analyze What You Discovered

After exploring the repos, you should have discovered:

1. **torch version** needed by SGLang (from pyproject.toml)
2. **vLLM version** and its dependencies (from vLLM's requirements.txt)
3. **pydantic version** - check if vLLM needs v1.x or v2.x
4. **xformers version** compatible with torch version
5. **flashinfer availability** - wheel or build from source
6. **sgl-kernel availability** - PyPI or build from source

**Write down your discoveries before proceeding.**

---

## Step 2: Verify Package Availability with Web Searches

### Check Flashinfer Wheels
```
WebFetch https://flashinfer.ai/whl/cu121/torch{TORCH_MAJOR}.{TORCH_MINOR}/
```
If empty/404, must build from source.

### Check sgl-kernel on PyPI
```
WebFetch https://pypi.org/simple/sgl-kernel/
```
Look for exact version. If not found, build from source.

### Check torch Wheel Exists
```
WebFetch https://download.pytorch.org/whl/cu{CUDA}/torch-{VERSION}/
```

### Check xformers Wheel
```
WebFetch https://download.pytorch.org/whl/cu{CUDA}/xformers/
```

---

## Step 3: Determine Base Image

**Based on torch version you discovered:**

- torch 2.1.x-2.3.x: `pytorch/pytorch:{version}-cuda12.1-cudnn8-devel`
- torch 2.4.x-2.5.x: `nvidia/cuda:12.1.1-devel-ubuntu20.04`
- torch 2.6.x+: `nvidia/cuda:12.4.1-devel-ubuntu22.04`

---

## Step 4: Handle vLLM Dependencies (The Hard Part)

### CRITICAL: vLLM will pull wrong torch if you install normally

```dockerfile
# WRONG - this lets vLLM pull whatever torch it wants:
RUN pip install vllm==0.4.2

# CORRECT - install vLLM without dependencies, then add deps manually:
RUN pip install vllm==0.4.2 --no-deps
```

### After --no-deps, install vLLM's deps from your exploration:

Use the actual requirements you found in vLLM's requirements.txt:

```dockerfile
# Install vLLM dependencies (from your exploration of vllm/requirements.txt)
RUN pip install \
    "sentencepiece" \
    "numpy" \
    "requests" \
    "py-cpuinfo" \
    "transformers>=4.40.0" \
    "tokenizers>=0.19.1" \
    "fastapi" \
    "uvicorn[standard]" \
    "nvidia-ml-py"
    # Add other deps you found in requirements.txt
```

### Handle pydantic Version Conflict

From your exploration of vLLM's requirements, check:
- Does vLLM require `pydantic>=2.0` or `pydantic>=1.10,<2.0`?
- If vLLM needs pydantic v2 but it conflicts with other packages, you may need to:
  - Remove optional packages that conflict (like outlines, lm-format-enforcer)
  - Or find alternative compatible versions

```dockerfile
# If vLLM requires pydantic v1.x (check requirements.txt!)
RUN pip install "pydantic>=1.10,<2.0"

# If pydantic v2, also need newer typing_extensions
RUN pip install "pydantic>=2.0" "typing_extensions>=4.12"
```

---

## Step 5: Handle Flashinfer

Based on your WebFetch to flashinfer.ai/whl/:

### If wheel exists:
```dockerfile
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu{CUDA}/torch{TORCH}/
```

### If no wheel (build from source):
```dockerfile
RUN pip install ninja numpy packaging && \
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout v0.2.6 && \
    cd python && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    rm -rf /tmp/flashinfer
```

---

## Step 6: Handle sgl-kernel

Based on your WebFetch to pypi.org/simple/sgl-kernel/:

### If version exists on PyPI:
```dockerfile
RUN pip install sgl-kernel=={VERSION}
```

### If not on PyPI (build from source):
```dockerfile
RUN git clone https://github.com/sgl-project/sgl-kernel.git /tmp/sgl-kernel && \
    cd /tmp/sgl-kernel && \
    git checkout v{VERSION} && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=96 pip install --no-build-isolation . && \
    rm -rf /tmp/sgl-kernel
```

---

## Step 7: Handle xformers

From your exploration, find the xformers version compatible with your torch:

```dockerfile
# Install xformers matching torch version (from your exploration)
RUN pip install xformers=={DISCOVERED_VERSION} --index-url https://download.pytorch.org/whl/cu{CUDA}
```

---

## Step 8: Install SGLang

### Patch pyproject.toml first (remove already-installed deps):

```dockerfile
RUN cd /sgl-workspace/sglang && \
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml && \
    # Clean up any empty commas left behind
    sed -i 's/,\s*,/,/g' python/pyproject.toml && \
    sed -i 's/\[,/[/g' python/pyproject.toml && \
    sed -i 's/,\]/]/g' python/pyproject.toml
```

### Install SGLang:

```dockerfile
# pyproject.toml is in python/ subdirectory
RUN pip install -e "python[all]"
```

---

## Step 9: HARDCODE the Commit SHA (3 occurrences)

Every Dockerfile must have the full 40-char SHA hardcoded in EXACTLY 3 places:

```dockerfile
# 1st occurrence: ENV
ENV SGLANG_COMMIT=<FULL_40_CHAR_SHA>

# 2nd occurrence: git checkout
RUN git clone https://github.com/sgl-project/sglang.git sglang && \
    cd sglang && \
    git checkout <FULL_40_CHAR_SHA>

# 3rd occurrence: verification
RUN cd /sgl-workspace/sglang && \
    ACTUAL=$(git rev-parse HEAD) && \
    EXPECTED="<FULL_40_CHAR_SHA>" && \
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH" && exit 1) && \
    echo "$ACTUAL" > /opt/sglang_commit.txt
```

---

## Step 10: Final Verification

```dockerfile
RUN python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import sglang; print('SGLang import OK')" && \
    python3 -c "import flashinfer; print('flashinfer OK')" 2>/dev/null || true && \
    python3 -c "import vllm; print('vLLM OK')" 2>/dev/null || true
```

---

## Build Settings

- **TORCH_CUDA_ARCH_LIST="9.0"** - H100 only (fastest build)
- **MAX_JOBS=96** - Use all 96 cores

---

## Troubleshooting During Exploration

### Can't find vLLM requirements.txt at that tag?
Try these paths:
- `requirements.txt`
- `requirements-cuda.txt`
- `requirements-common.txt`
- `pyproject.toml` (look in [project.dependencies])

### xformers version unclear?
WebSearch: "xformers {version} torch requirement github"
Or check xformers releases page for compatibility notes.

### pydantic conflict?
Check which packages need pydantic v1 vs v2:
- outlines, lm-format-enforcer often need pydantic v2
- older vLLM/fastapi may need pydantic v1
- Remove conflicting optional packages if needed

### flashinfer version not found?
Check available tags: `git tag | grep -E "^v[0-9]"` in flashinfer repo.
Use nearest available version that's compatible with your torch.

---

## Summary: The Discovery Process

1. **Clone repos** to temp dir with unique path
2. **Read SGLang pyproject.toml** at exact commit
3. **Clone vLLM** at required version, read its requirements
4. **WebSearch/WebFetch** for wheel availability
5. **Determine compatible versions** from actual repo contents
6. **Write Dockerfile** using discovered versions
7. **Patch pyproject.toml** to remove pre-installed deps
8. **Verify imports** work

**Never guess. Always explore. Trust the source code, not lookup tables.**

---

## Output Location

Write fixed Dockerfile to:
```
fixed-dockerfiles/{pr_date}/{short_hash}.Dockerfile
```

---

## Verification After Writing

After writing the Dockerfile:
1. Read it back
2. Count full SHA occurrences (must be exactly 3)
3. Verify base image matches discovered torch version
4. Verify all version pins came from your exploration
