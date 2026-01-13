#!/bin/bash
# Build and push Docker image for commit 6ea1e6ac (baseline for 1acca3a2)

set -e

COMMIT="6ea1e6ac6e2fa949cebd1b4338f9bfb7036d14fe"
COMMIT_SHORT="6ea1e6ac"
IMAGE="ayushnangia16/nvidia-sglang-docker:${COMMIT}"
DOCKERFILE="fixed-dockerfiles/2025-05-02/${COMMIT_SHORT}.Dockerfile"

echo "=========================================="
echo "Building SGLang Docker image"
echo "=========================================="
echo "Commit: ${COMMIT}"
echo "Dockerfile: ${DOCKERFILE}"
echo "Image: ${IMAGE}"
echo ""

cd /home/ubuntu/vllm-docker-build

# Build the image
echo "[1/2] Building Docker image (this will take a while - flashinfer compiles CUDA kernels)..."
docker build -f "${DOCKERFILE}" -t "${IMAGE}" .

echo ""
echo "[2/2] Pushing to Docker Hub..."
docker push "${IMAGE}"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Image: ${IMAGE}"
echo ""
echo "Now run the benchmark:"
echo "  cd /home/ubuntu/sglang-modal-benchmark"
echo "  python run_benchmark.py --commit 1acca3a2 --baseline-only"
