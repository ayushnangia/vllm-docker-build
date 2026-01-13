# SGLang Docker Build Guidelines

## Common Issues

### FlashInfer Not Found at Runtime

**Symptom:** Benchmark fails immediately with `No module named 'flashinfer'`

**Cause:** Using `pip install -e .` (editable install) then deleting the source directory.

**Fix:** Use `pip install .` (non-editable) instead:

```dockerfile
# WRONG - editable install breaks when source is deleted
RUN pip install -e . --no-build-isolation && \
    rm -rf /tmp/flashinfer

# CORRECT - installs to site-packages, source can be deleted
RUN pip install . --no-build-isolation && \
    rm -rf /tmp/flashinfer
```

**Always add verification:**
```dockerfile
RUN python3 -c "import flashinfer; print('FlashInfer installed successfully')"
```

### FlashInfer Build from Source

For torch 2.6+ / CUDA 12.4+, no prebuilt wheels exist. Build from source:

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="9.0"  # H100
ENV MAX_JOBS=96

WORKDIR /tmp
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && \
    git checkout v0.2.5 && \
    git submodule update --init --recursive && \
    MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install . --no-build-isolation && \
    cd / && rm -rf /tmp/flashinfer
```

## Build Commands

```bash
# Build image
docker build -f fixed-dockerfiles/2025-05-02/6ea1e6ac.Dockerfile \
    -t ayushnangia16/nvidia-sglang-docker:6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe .

# Push to Docker Hub
docker push ayushnangia16/nvidia-sglang-docker:6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe
```

## Verification

After building, verify inside container:

```bash
docker run --rm -it --gpus all <image> python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import flashinfer
print('FlashInfer: OK')

import sgl_kernel
print('sgl_kernel: OK')

import sglang
print(f'SGLang: {sglang.__version__}')
"
```

## Dockerfile Checklist

- [ ] Base image matches torch CUDA version (e.g., `nvidia/cuda:12.4.1-devel-ubuntu22.04` for torch+cu124)
- [ ] `TORCH_CUDA_ARCH_LIST` set for target GPU (9.0 for H100, 8.0 for A100)
- [ ] FlashInfer uses `pip install .` not `pip install -e .`
- [ ] FlashInfer import verification step added
- [ ] SGLang commit hash hardcoded in 3 places (ENV, git checkout, verification)
- [ ] sgl-kernel version compatible with SGLang version
