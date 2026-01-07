#!/bin/bash
# push_custom_benchmarks.sh - Build, Verify, and Push all 4 custom SGLang images

set -e

REPO_DIR="./sglang"
DOCKERHUB_USER="ayushnangia16"
IMAGE_NAME="nvidia-sglang-docker"

build_verify_push() {
    local commit=$1
    local dockerfile=$2
    local short=${commit:0:7}
    local type=$3
    
    echo "========================================================="
    echo "BUILDING [$type]: $short"
    echo "========================================================="

    # 1. Export clean source
    local build_ctx="/tmp/sglang_push_$short"
    rm -rf "$build_ctx"
    mkdir -p "$build_ctx"
    cd "$REPO_DIR"
    git archive "$commit" | tar -x -C "$build_ctx"
    cd - > /dev/null

    # 2. Build for GPU (amd64) and Load locally for verification
    cp "$dockerfile" "$build_ctx/Dockerfile"
    echo "Running Docker Buildx..."
    docker buildx build \
        --platform linux/amd64 \
        --build-arg SGLANG_COMMIT="$commit" \
        -t "$DOCKERHUB_USER/$IMAGE_NAME:$commit" \
        -f "$build_ctx/Dockerfile" \
        --load \
        "$build_ctx"

    # 3. Verify locally
    echo "Verifying $short..."
    if python3 verify_sglang_docker.py "$commit" --repo "$DOCKERHUB_USER/$IMAGE_NAME"; then
        echo "⭐ VERIFICATION PASSED for $short"
    else
        echo "❌ VERIFICATION FAILED for $short"
        exit 1
    fi

    # 4. Push to Docker Hub
    echo "Pushing $short to $DOCKERHUB_USER/$IMAGE_NAME..."
    docker push "$DOCKERHUB_USER/$IMAGE_NAME:$commit"
    
    echo "✅ Successfully pushed $short"
    rm -rf "$build_ctx"
    echo ""
}

echo "Starting build and push for 2 benchmark pairs (4 images total)..."
echo ""

# PAIR 1: Triton-based (April 2024)
build_verify_push "da19434c2f3cbe4f367f84993da0bcbd84efb6ba" "custom-dockerfiles/Dockerfile.da19434" "Triton Parent"
build_verify_push "9216b10678a036a1797e19693b0445c889016687" "custom-dockerfiles/Dockerfile.9216b10" "Triton Target"

# PAIR 2: Standard-based (Feb 2024)
build_verify_push "45d6592d4053fe8b2b8dc9440f64c900de040d09" "custom-dockerfiles/Dockerfile.45d6592" "Legacy Parent"
build_verify_push "bb3a3b6675b1844a13ebe368ad693f3dc75b315b" "custom-dockerfiles/Dockerfile.bb3a3b6" "Legacy Target"

echo "========================================================="
echo "ALL 4 IMAGES PUSHED TO ayushnangia16/nvidia-sglang-docker"
echo "========================================================="
