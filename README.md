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
- Re-run with the next batch by editing `commits.txt` (remove the processed top rows) or increasing `batch_size` as desired.

## Notes
- Builds do not require a GPU. Runtime does.
- Keep `--platform linux/amd64` for CUDA base images.
- If you hit any rate limits or transient failures, re-run; the cache will avoid re-downloading layers.
