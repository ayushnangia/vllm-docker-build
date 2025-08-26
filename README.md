# vllm-docker-build

This repo builds NVIDIA (main) vLLM Docker images for specific commits and pushes them to Docker Hub.

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
