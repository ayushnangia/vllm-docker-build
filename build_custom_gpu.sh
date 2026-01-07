#!/bin/bash
# build_custom_gpu.sh - Build custom SGLang images for GPU (parallel)

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="$SCRIPT_DIR/sglang"
DOCKERHUB_USER="ayushnangia16"
IMAGE_NAME="nvidia-sglang-docker"
LOG_DIR="/tmp/sglang_build_logs"
mkdir -p "$LOG_DIR"

build_commit() {
    local commit=$1
    local dockerfile=$2
    local short=${commit:0:7}
    local log_file="$LOG_DIR/${short}.log"

    {
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
        rm -rf "$build_ctx"
    } > "$log_file" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "✅ BUILD COMPLETE: $short (log: $log_file)"
    else
        echo "❌ BUILD FAILED: $short (log: $log_file)"
    fi
    return $status
}

verify_commit() {
    local commit=$1
    local short=${commit:0:7}

    echo "Verifying $short..."
    if python3 "$SCRIPT_DIR/verify_sglang_docker.py" "$commit" --repo "$DOCKERHUB_USER/$IMAGE_NAME"; then
        echo "⭐ VERIFICATION PASSED for $short"
        return 0
    else
        echo "❌ VERIFICATION FAILED for $short"
        return 1
    fi
}

echo "🚀 Starting parallel builds of 4 custom commits..."
echo ""

# Define all commits and their dockerfiles
declare -A COMMITS=(
    ["da19434c2f3cbe4f367f84993da0bcbd84efb6ba"]="$SCRIPT_DIR/custom-dockerfiles/Dockerfile.da19434"
    ["9216b10678a036a1797e19693b0445c889016687"]="$SCRIPT_DIR/custom-dockerfiles/Dockerfile.9216b10"
    ["45d6592d4053fe8b2b8dc9440f64c900de040d09"]="$SCRIPT_DIR/custom-dockerfiles/Dockerfile.45d6592"
    ["bb3a3b6675b1844a13ebe368ad693f3dc75b315b"]="$SCRIPT_DIR/custom-dockerfiles/Dockerfile.bb3a3b6"
)

# Start all builds in parallel
pids=()
for commit in "${!COMMITS[@]}"; do
    dockerfile="${COMMITS[$commit]}"
    build_commit "$commit" "$dockerfile" &
    pids+=($!)
    echo "Started build for ${commit:0:7} (PID: ${pids[-1]})"
done

echo ""
echo "⏳ Waiting for all ${#pids[@]} builds to complete..."
echo "   Logs in: $LOG_DIR/"
echo ""

# Wait for all builds and track failures
failed=()
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    if ! wait $pid; then
        failed+=($pid)
    fi
done

echo ""
if [ ${#failed[@]} -gt 0 ]; then
    echo "❌ ${#failed[@]} build(s) failed. Check logs in $LOG_DIR/"
    exit 1
fi

echo "✅ All builds completed successfully!"
echo ""

# Verify all images
echo "🔍 Verifying all images..."
verify_failed=0
for commit in "${!COMMITS[@]}"; do
    if ! verify_commit "$commit"; then
        verify_failed=1
    fi
done

echo ""
if [ $verify_failed -eq 1 ]; then
    echo "❌ Some verifications failed!"
    exit 1
fi

echo ""
echo "🎉 All custom GPU builds complete and verified!"
