#!/usr/bin/env python3
"""
Extract Dockerfiles from SGLang repo for all commits in the JSONL file.
Organizes by pr_date in custom-dockerfiles/{date}/{short_hash}.Dockerfile
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict

JSONL_PATH = Path("data/sglang-h100-32.jsonl")
REPO_PATH = Path("sglang.git")
OUTPUT_DIR = Path("custom-dockerfiles")

# Dockerfile locations to try (in order of preference)
DOCKERFILE_PATHS = [
    "docker/Dockerfile",
    "Dockerfile",
    "examples/usage/triton/Dockerfile",
]

def get_dockerfile_at_commit(repo_path: Path, commit: str) -> tuple[str | None, str | None]:
    """Try to extract Dockerfile content at a specific commit."""
    for dockerfile_path in DOCKERFILE_PATHS:
        result = subprocess.run(
            ["git", "show", f"{commit}:{dockerfile_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout, dockerfile_path
    return None, None

def main():
    # Parse JSONL and collect all commits with their dates
    commits_by_date = defaultdict(dict)  # date -> {short_hash: full_hash}

    with open(JSONL_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pr_date = obj["pr_date"]
            commit_hash = obj["commit_hash"]
            parent_commit = obj["parent_commit"]

            # Add both commit and parent
            commits_by_date[pr_date][commit_hash[:8]] = commit_hash
            commits_by_date[pr_date][parent_commit[:8]] = parent_commit

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Track results
    found = []
    missing = []

    # Extract Dockerfiles for each commit
    for date in sorted(commits_by_date.keys()):
        date_dir = OUTPUT_DIR / date
        date_dir.mkdir(exist_ok=True)

        for short_hash, full_hash in commits_by_date[date].items():
            content, source_path = get_dockerfile_at_commit(REPO_PATH, full_hash)

            if content:
                output_file = date_dir / f"{short_hash}.Dockerfile"
                output_file.write_text(content)
                found.append((date, short_hash, source_path))
                print(f"[OK] {date}/{short_hash} from {source_path}")
            else:
                missing.append((date, short_hash, full_hash))
                print(f"[MISSING] {date}/{short_hash} ({full_hash})")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Found: {len(found)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print(f"\nMissing commits (no Dockerfile found):")
        for date, short_hash, full_hash in missing:
            print(f"  {date}/{short_hash} ({full_hash})")

if __name__ == "__main__":
    main()
