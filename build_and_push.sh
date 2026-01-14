#!/bin/bash
# Build, tag, and push all SGLang Docker images
# Usage: ./build_and_push.sh [--no-push] [--filter PATTERN]

set -e

DOCKERHUB_USER="ayushnangia16"
REPO="nvidia-sglang-docker"
NO_PUSH=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-push) NO_PUSH=true; shift ;;
        --filter) FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Mapping of short SHA to full SHA (from data/sglang-h100-32.jsonl)
declare -A FULL_SHA=(
    # 2024-01-30
    ["6f560c76"]="6f560c761b2fc2f577682d0cfda62630f37a3bb0"
    ["cd687233"]="cd6872334e9ead684049b8fccd5f2dac9433b1b4"
    # 2024-02-03
    ["45d6592d"]="45d6592d4053fe8b2b8dc9440f64c900de040d09"
    ["bb3a3b66"]="bb3a3b6675b1844a13ebe368ad693f3dc75b315b"
    # 2024-04-17
    ["ca4f1ab8"]="ca4f1ab89c0c9bdd80fdfabcec52968fbde108bb"
    ["e822e590"]="e822e5900b98d89d19e0a293d9ad384f4df2945a"
    # 2024-04-25
    ["9216b106"]="9216b10678a036a1797e19693b0445c889016687"
    ["da19434c"]="da19434c2f3cbe4f367f84993da0bcbd84efb6ba"
    # 2024-05-11
    ["09deb20d"]="09deb20ddc67ffe1a8ca32e7b24c93a5bff0cee4"
    ["33b242df"]="33b242df9e98a1d1de4c3ee9b0082d97e40fee21"
    # 2024-07-03
    ["2a754e57"]="2a754e57b052e249ed4f8572cb6f0069ba6a495e"
    ["96c503eb"]="96c503eb6029d37f896e91466e23469378dfc3dc"
    # 2024-07-14
    ["564a898a"]="564a898ad975192b593be81387d11faf15cb1d3e"
    ["5d264a90"]="5d264a90ac5154d8e368ee558337dd3dd92e720b"
    # 2024-07-19
    ["ac971ff6"]="ac971ff633de330de3ded7f7475caaf7cd5bbdcd"
    ["e1792cca"]="e1792cca2491af86f29782a3b83533a6566ac75b"
    # 2024-08-09
    ["62757db6"]="62757db6f0f09a6dff15b1ee1ac3029602951509"
    ["73fa2d49"]="73fa2d49d539fd67548b0458a365528d3e3b6edc"
    # 2024-09-05
    ["62f15eea"]="62f15eea5a0b4266cdae965d0337fd33f6673736"
    ["ab4a83b2"]="ab4a83b25909aa98330b838a224e4fe5c943e483"
    # 2024-09-23
    ["2854a5ea"]="2854a5ea9fbb31165936f633ab99915dec760f8d"
    ["42a2d82b"]="42a2d82ba71dc86ca3b6342c978db450658b750c"
    # 2024-10-06
    ["58d1082e"]="58d1082e392cabbf26c404cb7ec18e4cb51b99e9"
    ["9c064bf7"]="9c064bf78af8558dbc50fbd809f65dcafd6fd965"
    ["c98e84c2"]="c98e84c21e4313d7d307425ca43e61753a53a9f7"
    # 2024-10-17
    ["5ab20cce"]="5ab20cceba227479bf5088a3fc95b1b4fe0ac3a9"
    ["b1709305"]="b170930534acbb9c1619a3c83670a839ceee763a"
    ["e5db40dc"]="e5db40dcbce67157e005f524bf6a5bea7dcb7f34"
    # 2024-10-21
    ["7ce36068"]="7ce36068914503c3a53ad7be23ab29831fb8aa63"
    ["efb099cd"]="efb099cdee90b9ad332fcda96d89dd91ddebe072"
    # 2024-10-23
    ["05b3bf5e"]="05b3bf5e8e4751cf51510198ae2e864c4b11ac2f"
    ["8f8f96a6"]="8f8f96a6217ea737c94e7429e480196319594459"
    # 2024-10-25
    ["30643fed"]="30643fed7f92be32540dfcdf9e4310e477ce0f6d"
    ["b77a02cd"]="b77a02cdfdb4cd58be3ebc6a66d076832c309cfc"
    # 2024-11-18
    ["9c745d07"]="9c745d078e29e153a64300bd07636c7c9c1c42d5"
    ["ebaa2f31"]="ebaa2f31996e80e4128b832d70f29f288b59944e"
    # 2025-01-19
    ["2bd18e2d"]="2bd18e2d767e3a0f8afb5aff427bc8e6e4d297c0"
    ["83452dbb"]="83452dbb4a19c6a2461e972eb2b64a2df9a466b8"
)

# Find all Dockerfiles
DOCKERFILES=$(find fixed-dockerfiles -name "*.Dockerfile" -type f | sort)

if [ -n "$FILTER" ]; then
    DOCKERFILES=$(echo "$DOCKERFILES" | grep "$FILTER")
fi

echo "=========================================="
echo "SGLang Docker Build Script"
echo "=========================================="
echo "User: $DOCKERHUB_USER"
echo "Repo: $REPO"
echo "Push: $([ "$NO_PUSH" = true ] && echo "DISABLED" || echo "ENABLED")"
echo ""

for dockerfile in $DOCKERFILES; do
    # Extract short SHA from filename
    short_sha=$(basename "$dockerfile" .Dockerfile)
    date_dir=$(dirname "$dockerfile" | xargs basename)

    # Get full SHA
    full_sha="${FULL_SHA[$short_sha]}"

    if [ -z "$full_sha" ]; then
        echo "WARNING: No full SHA mapping for $short_sha, skipping..."
        continue
    fi

    echo "=========================================="
    echo "Building: $dockerfile"
    echo "Short SHA: $short_sha"
    echo "Full SHA: $full_sha"
    echo "=========================================="

    # Build
    echo "[BUILD] docker build -f $dockerfile -t sglang:$short_sha ."
    docker build -f "$dockerfile" -t "sglang:$short_sha" .

    # Tag
    full_tag="${DOCKERHUB_USER}/${REPO}:${full_sha}"
    echo "[TAG] docker tag sglang:$short_sha $full_tag"
    docker tag "sglang:$short_sha" "$full_tag"

    # Push
    if [ "$NO_PUSH" = false ]; then
        echo "[PUSH] docker push $full_tag"
        docker push "$full_tag"
    else
        echo "[SKIP] Push disabled"
    fi

    echo ""
done

echo "=========================================="
echo "Build complete!"
echo "=========================================="
