# vllm-docker-build

This repo builds Docker images for SGLang/vLLM specific commits and pushes them to Docker Hub for benchmarking.

## What it does
- Reads `commits.txt` (one commit SHA per line)
- For each commit, prefers `docker/Dockerfile`, falling back to root `Dockerfile`
- Builds for `linux/amd64` and pushes `docker.io/<DOCKERHUB_USERNAME>/vllm:<commit>`
- Uses GitHub Actions cache to speed up subsequent builds

## Setup
1) Create a new GitHub repository and push this folder's contents to it.
2) In the repository Settings → Secrets and variables → Actions, add:
   - `DOCKERHUB_USERNAME`: your Docker Hub username
   - `DOCKERHUB_TOKEN`: a Docker Hub access token
3) Put your commit SHAs in `commits.txt` (already generated here from your dataset).

## Run
- Go to Actions → "Build vLLM Docker (NVIDIA)" → Run workflow.
- Choose a small `batch_size` (e.g., 10–20) to start.
- Control concurrency with `max_parallel` (start with 2–3).
- Already-pushed tags on Docker Hub and SHAs in `blacklist.txt` are skipped automatically.
- Re-run with the next batch by increasing `batch_size` as desired.

## Notes
- Builds do not require a GPU. Runtime does.
- Keep `--platform linux/amd64` for CUDA base images.
- If you hit any rate limits or transient failures, re-run; the cache will avoid re-downloading layers.

## Local usage

You can build locally using `local_build.py` with the same filtering (blacklist and skip-pushed):

```bash
python3 local_build.py \
  --dockerhub-username YOUR_NAME \
  --batch-size 20 \
  --max-parallel 2 \
  --skip-pushed
```

Flags:
- `--no-push` to test builds without pushing images.
- `--dataset` to point to a different JSONL.
- `--blacklist` to specify a different blacklist file.

## Fixed Dockerfiles

The `fixed-dockerfiles/` directory contains manually fixed Dockerfiles organized by PR date:

```
fixed-dockerfiles/
├── 2024-01-30/
│   ├── 6f560c76.Dockerfile
│   └── cd687233.Dockerfile
├── 2024-11-18/
│   ├── 9c745d07.Dockerfile
│   └── ebaa2f31.Dockerfile
└── ...
```

Each Dockerfile:
- Uses `pytorch/pytorch` base image (avoids Python build issues)
- Has verified dependency versions from PyPI
- Contains commit SHA in 3 places (ENV, git checkout, verification)
- Uses `ENTRYPOINT ["/bin/bash"]`

## Batch Build Script

Use `build_and_push.sh` to build, tag, and push multiple images:

```bash
# Build and push all fixed Dockerfiles
./build_and_push.sh

# Build only (no push) - for testing
./build_and_push.sh --no-push

# Filter by date directory
./build_and_push.sh --filter 2024-11-18

# Filter by commit SHA
./build_and_push.sh --filter 9c745d07
```

## Manual Build

To build a single image:

```bash
# Build
docker build -f fixed-dockerfiles/2024-11-18/9c745d07.Dockerfile -t sglang:9c745d07 .

# Tag with full SHA
docker tag sglang:9c745d07 ayushnangia16/nvidia-sglang-docker:9c745d078e29e153a64300bd07636c7c9c1c42d5

# Push
docker push ayushnangia16/nvidia-sglang-docker:9c745d078e29e153a64300bd07636c7c9c1c42d5
```

## Verify Docker Image

```bash
# Check commit proof
docker run --rm <image> cat /opt/sglang_commit.txt

# Test imports
docker run --rm <image> python3 -c "import sglang; import vllm; print('OK')"
```
