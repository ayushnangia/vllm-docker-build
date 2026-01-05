#!/usr/bin/env python3
"""
Detect Dockerfile paths in SGLang commits.

This script scans each commit in the sglang-docker.jsonl file and identifies
what Dockerfiles exist at that commit. This helps determine the correct
Dockerfile to use for each commit instead of using a fallback/generated one.
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

# Common Dockerfile locations to check
DOCKERFILE_PATTERNS = [
    "docker/Dockerfile",
    "Dockerfile",
    ".devcontainer/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "examples/usage/triton/Dockerfile",  # Older path
    "sgl-kernel/Dockerfile",
    "docker/*.Dockerfile",  # Catch all specialized Dockerfiles
]

def get_files_at_commit(repo_path: str, commit: str) -> list[str]:
    """Get list of all files at a specific commit using git ls-tree."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return []
        return result.stdout.strip().split("\n")
    except Exception as e:
        print(f"  Error listing files for {commit[:8]}: {e}", file=sys.stderr)
        return []


def find_dockerfiles_at_commit(repo_path: str, commit: str) -> list[str]:
    """Find all Dockerfile paths at a specific commit."""
    files = get_files_at_commit(repo_path, commit)
    dockerfiles = []

    for f in files:
        filename = Path(f).name
        # Match Dockerfile or *.Dockerfile or *.dockerfile
        if filename == "Dockerfile" or filename.lower().endswith(".dockerfile"):
            dockerfiles.append(f)

    return sorted(dockerfiles)


def get_dockerfile_content_preview(repo_path: str, commit: str, dockerfile_path: str, lines: int = 20) -> str:
    """Get first N lines of a Dockerfile at a specific commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{dockerfile_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return ""
        content_lines = result.stdout.split("\n")[:lines]
        return "\n".join(content_lines)
    except Exception:
        return ""


def check_flashinfer_in_dockerfile(repo_path: str, commit: str, dockerfile_path: str) -> dict:
    """Check if a Dockerfile mentions flashinfer and extract version info."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{dockerfile_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {"has_flashinfer": False}

        content = result.stdout
        info = {"has_flashinfer": "flashinfer" in content.lower()}

        # Look for flashinfer version patterns
        import re

        # Pattern: flashinfer==X.X.X or flashinfer>=X.X.X
        version_match = re.search(r'flashinfer[=<>]+([0-9.]+)', content, re.IGNORECASE)
        if version_match:
            info["flashinfer_version"] = version_match.group(1)

        # Pattern: flashinfer.ai/whl/cuXXX/torchX.X/
        whl_match = re.search(r'flashinfer\.ai/whl/(cu\d+)/(torch[0-9.]+)', content)
        if whl_match:
            info["flashinfer_cuda"] = whl_match.group(1)
            info["flashinfer_torch"] = whl_match.group(2)

        return info
    except Exception:
        return {"has_flashinfer": False}


def main():
    repo_path = "/home/ubuntu/vllm-docker-build/sglang"
    jsonl_path = "/home/ubuntu/vllm-docker-build/sglang-docker.jsonl"

    # Read commits from JSONL
    commits = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                commits.append(data["commit"])

    print(f"Scanning {len(commits)} commits for Dockerfiles...\n")

    # Track results
    results = []
    dockerfile_locations = defaultdict(list)  # dockerfile_path -> [commits]
    commits_without_dockerfiles = []

    for i, commit in enumerate(commits, 1):
        short_commit = commit[:8]
        print(f"[{i}/{len(commits)}] Checking {short_commit}...", end=" ")

        dockerfiles = find_dockerfiles_at_commit(repo_path, commit)

        if not dockerfiles:
            print("NO DOCKERFILES FOUND")
            commits_without_dockerfiles.append(commit)
            results.append({
                "commit": commit,
                "dockerfiles": [],
                "recommended": None
            })
            continue

        # Find the main Dockerfile (prefer docker/Dockerfile)
        recommended = None
        for preferred in ["docker/Dockerfile", "Dockerfile"]:
            if preferred in dockerfiles:
                recommended = preferred
                break
        if not recommended and dockerfiles:
            recommended = dockerfiles[0]

        # Check for flashinfer in recommended Dockerfile
        flashinfer_info = {}
        if recommended:
            flashinfer_info = check_flashinfer_in_dockerfile(repo_path, commit, recommended)

        print(f"Found {len(dockerfiles)} Dockerfile(s): {', '.join(dockerfiles)}")
        if flashinfer_info.get("has_flashinfer"):
            fi_details = []
            if "flashinfer_version" in flashinfer_info:
                fi_details.append(f"v{flashinfer_info['flashinfer_version']}")
            if "flashinfer_cuda" in flashinfer_info:
                fi_details.append(f"{flashinfer_info['flashinfer_cuda']}/{flashinfer_info['flashinfer_torch']}")
            if fi_details:
                print(f"         flashinfer: {', '.join(fi_details)}")

        for df in dockerfiles:
            dockerfile_locations[df].append(commit)

        results.append({
            "commit": commit,
            "dockerfiles": dockerfiles,
            "recommended": recommended,
            "flashinfer_info": flashinfer_info
        })

    # Summary report
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal commits scanned: {len(commits)}")
    print(f"Commits with Dockerfiles: {len(commits) - len(commits_without_dockerfiles)}")
    print(f"Commits WITHOUT Dockerfiles: {len(commits_without_dockerfiles)}")

    if commits_without_dockerfiles:
        print(f"\nCommits missing Dockerfiles ({len(commits_without_dockerfiles)}):")
        for c in commits_without_dockerfiles:
            print(f"  - {c[:12]}")

    print("\nDockerfile locations found:")
    for path, commit_list in sorted(dockerfile_locations.items(), key=lambda x: -len(x[1])):
        print(f"  {path}: {len(commit_list)} commits")

    # Write detailed results to JSON
    output_path = "/home/ubuntu/vllm-docker-build/dockerfile_detection_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_commits": len(commits),
            "commits_with_dockerfiles": len(commits) - len(commits_without_dockerfiles),
            "commits_without_dockerfiles": commits_without_dockerfiles,
            "dockerfile_locations": {k: len(v) for k, v in dockerfile_locations.items()},
            "detailed_results": results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")

    # Generate updated JSONL with Dockerfile paths
    updated_jsonl_path = "/home/ubuntu/vllm-docker-build/sglang-docker-with-paths.jsonl"
    with open(updated_jsonl_path, "w") as f:
        for result in results:
            entry = {
                "commit": result["commit"],
                "Dockerfile": result["recommended"],
                "all_dockerfiles": result["dockerfiles"]
            }
            if "flashinfer_info" in result:
                entry["flashinfer_info"] = result["flashinfer_info"]
            f.write(json.dumps(entry) + "\n")

    print(f"Updated JSONL saved to: {updated_jsonl_path}")


if __name__ == "__main__":
    main()
