#!/bin/bash
# build_triton_pair.sh - Build and Push Triton-based SGLang pair

set -e

REPO_DIR="./sglang"
DOCKERHUB_USER="ayushnangia16"
IMAGE_NAME="nvidia-sglang-docker"

build_and_push() {
    local commit=$1
    local dockerfile=$2
    local short=${commit:0:7}
    
    echo "========================================================="
    echo "BUILDING & PUSHING: $short (Triton Base)"
    echo "========================================================="

    # 1. Export clean source
    local build_ctx="/tmp/sglang_triton_$short"
    rm -rf "$build_ctx"
    mkdir -p "$build_ctx"
    cd "$REPO_DIR"
    git archive "$commit" | tar -x -C "$build_ctx"
    cd - > /dev/null

    # 2. Build for GPU (amd64) and Load locally for verification
    cp "$dockerfile" "$build_ctx/Dockerfile"
    docker buildx build \
        --platform linux/amd64 \
        --build-arg SGLANG_COMMIT="$commit" \
        -t "$DOCKERHUB_USER/$IMAGE_NAME:$commit" \
        -f "$build_ctx/Dockerfile" \
        --load \
        "$build_ctx"

    # 3. Verify
    echo "Verifying $short..."
    if python3 verify_sglang_docker.py "$commit" --repo "$DOCKERHUB_USER/$IMAGE_NAME"; then
        echo "⭐ VERIFICATION PASSED for $short"
    else
        echo "❌ VERIFICATION FAILED for $short"
        exit 1
    fi

    # 4. Push to Docker Hub
    echo "Pushing $short to Docker Hub..."
    docker push "$DOCKERHUB_USER/$IMAGE_NAME:$commit"
    
    echo "✅ Successfully pushed $short"
    rm -rf "$build_ctx"
}

# 1. Build & Push Parent (Baseline)
build_and_push "da19434c2f3cbe4f367f84993da0bcbd84efb6ba" "custom-dockerfiles/Dockerfile.da19434"

# 2. Build & Push Child (Target)
build_and_push "9216b10678a036a1797e19693b0445c889016687" "custom-dockerfiles/Dockerfile.9216b10"

echo ""
echo "Triton pair build and push complete!"
