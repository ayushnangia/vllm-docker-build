# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository builds Docker images for specific commits from ML inference frameworks (vLLM, SGLang) and pushes them to Docker Hub. Primary use case: benchmarking performance changes between commits by ensuring each Docker image contains exactly the specified commit.

## Build Commands

### Local Build (Primary)
```bash
python3 local_build.py \
  --dockerhub-username YOUR_NAME \
  --batch-size 20 \
  --max-parallel 2 \
  --skip-pushed \
  --project sglang  # or vllm
```

Key flags:
- `--no-push`: Test builds locally without pushing to Docker Hub
- `--skip-pushed`: Skip commits already on Docker Hub
- `--show-build-logs`: Stream build logs with `[commit]` prefixes
- `--dataset`: JSONL file with commits (default: `nvidia-sglang-docker.jsonl`)
- `--blacklist`: Skip commits in this file (default: `blacklist.txt`)
- `--project`: `sglang` (default) or `vllm`

### Docker Image Verification
```bash
# Verify single commit
python3 verify_sglang_docker.py <commit_sha>

# Verify all commits from CSV files
python3 verify_sglang_docker.py --all

# JSON output
python3 verify_sglang_docker.py --all --json
```

### Dockerfile Detection
```bash
python3 detect_dockerfiles.py
```
Scans commits in JSONL to find what Dockerfiles exist at each commit.

### GitHub Actions
Trigger: Actions → "Build vLLM Docker (NVIDIA)" → Run workflow with `batch_size` and `max_parallel`.

## Architecture

### Build Pipeline (`local_build.py`)

1. **Commit extraction**: Reads commits from JSONL dataset, filters by blacklist and existing tags
2. **Context materialization**: Uses `git archive` to export clean build context for each commit (thread-safe via `_repo_lock`)
3. **Dockerfile resolution**: Checks `docker/Dockerfile`, then `Dockerfile`, then `examples/usage/triton/Dockerfile`
4. **Automatic fixes** (`_apply_generic_dockerfile_fixes`):
   - Injects `ARG SGLANG_COMMIT` and `/opt/sglang_commit.txt` for commit verification
   - Replaces `git clone` with `COPY` to use archived build context
   - Builds Python 3.10 from source on Ubuntu 20.04 (deadsnakes PPA deprecated)
   - Fixes DeepEP/nvshmem compilation issues
   - Normalizes Dockerfile syntax (uppercase `AS`, `ENV key=value`)
5. **Build execution**: Parallel builds via ThreadPoolExecutor with Buildx caching

### Special Commit Handling

Some commits need dependencies built from source:
- `FLASHINFER_FROM_SOURCE_COMMITS`: Commits needing flashinfer built from source (no prebuilt wheels)
- `SGL_KERNEL_FROM_SOURCE_COMMITS`: Commits needing sgl-kernel built from source

### Key Data Files

| File | Purpose |
|------|---------|
| `nvidia-sglang-docker.jsonl` | SGLang commits with Dockerfile content |
| `nvidia-vllm-docker.jsonl` | vLLM commits with Dockerfile content |
| `blacklist.txt` / `blacklist-sglang.txt` | Commits to skip |
| `commit-status/success_with_dockerfile.csv` | Commits with successful benchmarks |
| `commit-status/other_commits.csv` | Parent commits for comparison |
| `commit-status/failed/*.txt` | Categorized failure lists by error type |

### Commit Verification System

**Why it matters**: When benchmarking `commit_hash` vs `parent_commit` (often 1 commit apart), the wrong code in Docker means invalid benchmarks.

Verification checks:
1. Image exists locally
2. `/opt/sglang_commit.txt` matches expected SHA
3. SGLang installed via pip (editable install preferred)
4. `import sglang` succeeds

Build-time commit identity:
- `--label org.opencontainers.image.revision=<commit>`
- `--build-arg SGLANG_COMMIT=<commit>`
- Writes SHA to `/opt/sglang_commit.txt`

### Manual Docker Verification
```bash
# Check labels
docker inspect ayushnangia16/nvidia-sglang-docker:<commit> | jq '.[0].Config.Labels'

# Check commit proof
docker run --rm ayushnangia16/nvidia-sglang-docker:<commit> cat /opt/sglang_commit.txt

# Check pip install type
docker run --rm ayushnangia16/nvidia-sglang-docker:<commit> pip show sglang | grep -E "^(Version|Location|Editable)"

# Test import
docker run --rm ayushnangia16/nvidia-sglang-docker:<commit> python3 -c "import sglang; print('OK')"
```

## Required Secrets (GitHub Actions)

- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

## Dependencies

- Docker with Buildx enabled
- git
- Python 3 (optional: `tqdm` for progress bars)
