#!/bin/bash
# Rebuild remaining 9 commits with flashinfer fix and push

COMMITS=(
  "7ce36068914503c3a53ad7be23ab29831fb8aa63"
  "9c064bf78af8558dbc50fbd809f65dcafd6fd965"
  "9c745d078e29e153a64300bd07636c7c9c1c42d5"
  "ab4a83b25909aa98330b838a224e4fe5c943e483"
  "b170930534acbb9c1619a3c83670a839ceee763a"
  "b77a02cdfdb4cd58be3ebc6a66d076832c309cfc"
  "c98e84c21e4313d7d307425ca43e61753a53a9f7"
  "dc67d9769382cf83b3e2644a4366d6473445a6c6"
  "e5db40dcbce67157e005f524bf6a5bea7dcb7f34"
)

cd /home/ubuntu/vllm-docker-build

for commit in "${COMMITS[@]}"; do
  short="${commit:0:8}"
  echo "=========================================="
  echo "Building $short..."
  echo "=========================================="

  # Create single commit dataset
  echo "{\"commit\": \"$commit\", \"Dockerfile\": \"docker/Dockerfile\"}" > /tmp/single_commit.jsonl

  # Build (no push)
  python3 local_build.py \
    --dockerhub-username ayushnangia16 \
    --image-name nvidia-sglang-docker \
    --dataset /tmp/single_commit.jsonl \
    --workdir sglang \
    --repo-url https://github.com/sgl-project/sglang.git \
    --project sglang \
    --batch-size 1 \
    --max-parallel 1 \
    --no-push

  # Push manually
  echo "Pushing $short..."
  docker push "ayushnangia16/nvidia-sglang-docker:$commit"

  echo "$short DONE"
  echo ""
done

echo "All 9 commits rebuilt and pushed!"
