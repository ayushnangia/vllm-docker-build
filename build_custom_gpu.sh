#!/bin/bash
# build_custom_gpu.sh - Build custom SGLang images for GPU from Mac

set -e

REPO_DIR="./sglang"
DOCKERHUB_USER="ayushnangia16"
IMAGE_NAME="nvidia-sglang-docker"

build_commit() {
    local commit=$1
    local dockerfile=$2
    local short=${commit:0:7}
    
    echo "========================================================="
    echo "BUILDING: $short for GPU (linux/amd64)"
    echo "========================================================="

    # 1. Create a clean temporary directory for the build context
    local build_ctx="/tmp/sglang_build_$short"
    rm -rf "$build_ctx"
    mkdir -p "$build_ctx"

    # 2. Export ONLY the files from that specific commit
    echo "Exporting source for commit $commit..."
    cd "$REPO_DIR"
    git archive "$commit" | tar -x -C "$build_ctx"
    cd - > /dev/null

    # 3. Copy the custom Dockerfile into the context
    cp "$dockerfile" "$build_ctx/Dockerfile"

    # 4. Build for linux/amd64 (GPU system)
    echo "Starting Docker Buildx for linux/amd64..."
    docker buildx build \
        --platform linux/amd64 \
        --build-arg SGLANG_COMMIT="$commit" \
        -t "$DOCKERHUB_USER/$IMAGE_NAME:$commit" \
        -f "$build_ctx/Dockerfile" \
        --load \
        "$build_ctx"

    echo "✅ Successfully built $DOCKERHUB_USER/$IMAGE_NAME:$commit"

    # 5. Verify the image locally before finishing
    echo "Running verification for $short..."
    if python3 verify_sglang_docker.py "$commit" --repo "$DOCKERHUB_USER/$IMAGE_NAME"; then
        echo "⭐ VERIFICATION PASSED for $short"
    else
        echo "❌ VERIFICATION FAILED for $short"
        exit 1
    fi

    rm -rf "$build_ctx"
}

# Build the custom commits and their parents
# 1. 9216b10 and its parent da19434
build_commit "da19434c2f3cbe4f367f84993da0bcbd84efb6ba" "custom-dockerfiles/Dockerfile.da19434"
build_commit "9216b10678a036a1797e19693b0445c889016687" "custom-dockerfiles/Dockerfile.9216b10"

# 2. bb3a3b6 and its parent 45d6592
build_commit "45d6592d4053fe8b2b8dc9440f64c900de040d09" "custom-dockerfiles/Dockerfile.45d6592"
build_commit "bb3a3b6675b1844a13ebe368ad693f3dc75b315b" "custom-dockerfiles/Dockerfile.bb3a3b6"

echo ""
echo "All custom GPU builds complete!"
