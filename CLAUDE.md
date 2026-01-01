# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository builds NVIDIA vLLM Docker images for specific commits from the vllm-project/vllm repository and pushes them to Docker Hub. It supports both GitHub Actions (CI) and local builds.

## Build Commands

### Local Build
```bash
python3 local_build.py \
  --dockerhub-username YOUR_NAME \
  --batch-size 20 \
  --max-parallel 2 \
  --skip-pushed
```

Key flags:
- `--no-push`: Test builds without pushing to Docker Hub
- `--skip-pushed`: Skip commits that already have tags on Docker Hub
- `--show-build-logs`: Stream build logs with commit prefixes
- `--dataset`: Point to a different JSONL file (default: `nvidia-vllm-docker.jsonl`)
- `--blacklist`: Specify a different blacklist file (default: `blacklist.txt`)
- `--platform`: Build platform (default: `linux/amd64`)

### GitHub Actions
Trigger via Actions → "Build vLLM Docker (NVIDIA)" → Run workflow. Configure `batch_size` and `max_parallel` inputs.

## Architecture

### Core Components

**`local_build.py`**: Main local build orchestrator
- Reads commits from `nvidia-vllm-docker.jsonl` (JSONL with commit SHAs and optional Dockerfile content)
- Filters out commits in `blacklist.txt` and optionally those already pushed to Docker Hub
- Uses `git archive` to export clean build contexts (avoids worktree race conditions)
- Applies automatic fixes to Dockerfiles before building (outlines pinning, setuptools-scm env vars, Dockerfile syntax normalization)
- Supports parallel builds via ThreadPoolExecutor
- Uses local Buildx cache for layer sharing across builds

**`.github/workflows/build-vllm.yml`**: CI workflow
- Matrix strategy builds commits in parallel
- Uses GHA cache for Docker layers
- Maximizes disk space on runners before building

### Key Data Files

- `nvidia-vllm-docker.jsonl`: Source dataset with commit SHAs and Dockerfile content
- `blacklist.txt`: Commit SHAs to skip (one per line)
- `commits.txt`: Simple list of commit SHAs (legacy/alternative format)

### Build Context Fixes (`_apply_context_fixes`)

The build system automatically patches known issues before building:
- Pins `outlines` package to `<0.0.43` to avoid missing `pyairports` dependency
- Removes problematic deps like `sparsezoo`/`sparseml`
- Sets `SETUPTOOLS_SCM_PRETEND_VERSION` env vars for builds without `.git`
- Normalizes Dockerfile syntax (uppercase `AS`, `ENV` key=value format)
- Removes `.git` bind mounts that fail in archived contexts

## Required Secrets (GitHub Actions)

- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

## Dependencies

- Docker with Buildx enabled
- git
- Python 3 with optional `tqdm` for progress bars
