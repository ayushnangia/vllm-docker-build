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

---

## SGLang Commit Tracking & Verification

### Primary Focus Files

We only care about these files for SGLang benchmarking:

- **`sglang_track_all.json`**: Master tracking file with all SGLang commits (JSONL format)
- **`commit-status/success.txt`**: 16 commits with successful benchmarks
- **`commit-status/failed/`**: Categorized failure lists by error type

### Commit Verification Workflow

For each commit, we do **case-by-case analysis** of:

1. **`commit_hash`** (human/PR commit) - the performance change we're measuring
2. **`parent_commit`** (baseline) - the commit before the PR, used for comparison

For BOTH commits, verify:
- Dockerfile exists and is correct (`has_dockerfile`, `dockerfile_path`, `dockerfile_content`)
- The right commit is compiled into the Docker image
- The build is pushed correctly to Docker Hub (`ayushnangia16/nvidia-sglang-docker:<commit_hash>`)

### Commit Identity Inside Docker

**CRITICAL**: Always embed commit identifiers in Docker images so we can verify the correct code is running:

1. **Build-time label**: `--label org.opencontainers.image.revision=<commit_hash>`
2. **Build-arg**: `--build-arg SGLANG_COMMIT=<commit_hash>`
3. **Runtime proof file**: `/opt/sglang_commit.txt` containing the commit SHA
4. **Dockerfile ARG**: `ARG SGLANG_COMMIT` injected after first FROM

### Verification Script

Use `verify_sglang_docker.py` to validate Docker images have correct commit-level installation:

```bash
# Verify a single commit
python3 verify_sglang_docker.py 6b231325b9782555eb8e1cfcf27820003a98382b

# Verify all commits from success_with_dockerfile.txt
python3 verify_sglang_docker.py --all

# Output as JSON
python3 verify_sglang_docker.py --all --json
```

The script checks:
1. **Image exists** locally
2. **Commit file** `/opt/sglang_commit.txt` exists and matches expected SHA
3. **SGLang installed** via pip (checks for editable install)
4. **SGLang imports** successfully

### Manual Verification Commands

```bash
# Check Docker image labels
docker inspect ayushnangia16/nvidia-sglang-docker:<commit> | jq '.[0].Config.Labels'

# Check commit proof inside container
docker run --rm ayushnangia16/nvidia-sglang-docker:<commit> cat /opt/sglang_commit.txt

# Verify sglang is installed from local source (editable)
docker run --rm ayushnangia16/nvidia-sglang-docker:<commit> pip show sglang | grep -E "^(Version|Location|Editable)"

# Test sglang import
docker run --rm ayushnangia16/nvidia-sglang-docker:<commit> python3 -c "import sglang; print('OK')"
```

### Why This Matters

Since `parent_commit` and `commit_hash` are often just **1 commit apart**, if the Docker build doesn't properly pin to the exact commit, we could benchmark the WRONG code and have NO way to detect it. The verification workflow ensures:

- Human commit image contains exactly the PR code
- Parent commit image contains exactly the baseline code
- Benchmarks are comparing the correct versions
