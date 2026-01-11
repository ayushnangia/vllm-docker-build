#!/usr/bin/env python3
"""
Automated Dockerfile fixer using Claude Code CLI.
Runs multiple instances in parallel with extended thinking enabled.

Usage:
    python3 fix_all_dockerfiles.py                    # Run with defaults (10 parallel)
    python3 fix_all_dockerfiles.py --parallel 5      # Run 5 at a time
    python3 fix_all_dockerfiles.py --dry-run         # Just show what would run
    python3 fix_all_dockerfiles.py --commit abc123   # Fix single commit
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

JSONL_PATH = Path("data/sglang-h100-32.jsonl")
DOCKERFILES_DIR = Path("custom-dockerfiles")
SGLANG_REPO = Path("sglang.git")
FIXED_DIR = Path("fixed-dockerfiles")
LOG_DIR = Path("fix-logs")
GUIDELINES_PATH = Path("DOCKERFILE_FIX_GUIDELINES.md")

# Defaults (can be overridden via CLI)
DEFAULT_PARALLEL = 10
DEFAULT_TIMEOUT = 1800  # 30 minutes

# Thread-safe print
print_lock = threading.Lock()
def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


def get_pyproject_toml(commit_sha: str) -> str:
    """Extract pyproject.toml content at a specific commit."""
    result = subprocess.run(
        ["git", "show", f"{commit_sha}:python/pyproject.toml"],
        cwd=SGLANG_REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"ERROR: Could not extract pyproject.toml: {result.stderr}"
    return result.stdout


def get_dockerfile_path(commit_sha: str, pr_date: str) -> Path:
    """Find the Dockerfile for a commit."""
    short_hash = commit_sha[:8]
    return DOCKERFILES_DIR / pr_date / f"{short_hash}.Dockerfile"


def build_claude_prompt(commit_sha: str, pr_date: str, pyproject_content: str, dockerfile_content: str) -> str:
    """Build the prompt for Claude."""
    short_hash = commit_sha[:8]
    return f'''You are fixing an SGLang Dockerfile for commit {commit_sha} (date: {pr_date}).

## CORE PRINCIPLE: DISCOVER EVERYTHING VIA WEB SEARCHES

**NEVER guess versions. NEVER use hardcoded tables.**
You MUST use WebFetch and WebSearch extensively to discover:
1. What versions existed at the commit date
2. What each package's dependencies are
3. Whether packages are compatible with each other

---

## STEP 0: DISCOVER DEPENDENCY REQUIREMENTS FROM PYPI (MANDATORY)

For EACH key package (outlines, pydantic, fastapi, typing_extensions), you MUST:

### 0a. Check PyPI release history to find versions from the commit era
```
WebFetch https://pypi.org/project/outlines/#history
WebFetch https://pypi.org/project/pydantic/#history
WebFetch https://pypi.org/project/fastapi/#history
WebFetch https://pypi.org/project/typing-extensions/#history
```

### 0b. CRITICAL: Check what pydantic version outlines requires
For example, if pyproject.toml says outlines==0.0.39:
```
WebFetch https://pypi.org/project/outlines/0.0.39/
```
Look at the "Requires" section - does it need pydantic v1 or v2?

### 0c. Check for known breaking changes
```
WebSearch "fastapi pydantic v2 migration version"
WebSearch "fastapi 0.126 pydantic requirement"
WebSearch "typing_extensions Sentinel version"
```

Key facts to discover:
- FastAPI >= 0.126.0 FORCES pydantic v2 (drops v1 support)
- typing_extensions.Sentinel was added in version 4.14.0 (June 2025)
- If you see Sentinel errors, your typing_extensions is too new

### 0d. Find the RIGHT pydantic version for the era
If outlines needs pydantic v2, find pydantic 2.x from that era:
```
WebFetch https://pypi.org/project/pydantic/2.7.1/
```
Check release date - pydantic 2.7.1 is April 23, 2024.

---

## STEP 1: EXPLORE ACTUAL REPOS

### 1a. Setup temp directory
```bash
WORK_DIR="/tmp/explore-{short_hash}"
rm -rf "$WORK_DIR" && mkdir -p "$WORK_DIR" && cd "$WORK_DIR"
```

### 1b. Clone SGLang at this commit
```bash
git clone https://github.com/sgl-project/sglang.git sglang
cd sglang && git checkout {commit_sha}
cat python/pyproject.toml
```

### 1c. Clone vLLM at required version
```bash
cd "$WORK_DIR"
# Use version from SGLang's pyproject.toml
git clone --depth 1 --branch v<VERSION> https://github.com/vllm-project/vllm.git vllm
cat vllm/requirements*.txt
cat vllm/pyproject.toml | head -100
```

### 1d. Cleanup
```bash
rm -rf "$WORK_DIR"
```

---

## STEP 2: CHECK WHEEL AVAILABILITY

### flashinfer wheels
```
WebFetch https://flashinfer.ai/whl/cu121/torch2.3/
```
If empty/404 → BUILD FROM SOURCE

### sgl-kernel on PyPI
```
WebFetch https://pypi.org/simple/sgl-kernel/
```
If version not found → BUILD FROM SOURCE

---

## STEP 3: BUILD CONSTRAINTS FILE WITH DISCOVERED VERSIONS

Create a constraints file with versions YOU discovered from PyPI:

```dockerfile
RUN cat > /opt/constraints.txt <<'EOF'
# Versions discovered from PyPI for {pr_date} era
fastapi==<VERSION_YOU_FOUND_ON_PYPI>
uvicorn==<VERSION_YOU_FOUND_ON_PYPI>
pydantic==<VERSION_YOU_FOUND_ON_PYPI>  # v1 or v2 based on outlines requirement!
typing_extensions==<VERSION_YOU_FOUND_ON_PYPI>  # BEFORE 4.14 to avoid Sentinel
outlines==<VERSION_FROM_PYPROJECT>
pyzmq==<VERSION_YOU_FOUND_ON_PYPI>
EOF
```

---

## STEP 4: WRITE DOCKERFILE

### Base image:
- torch 2.1.x-2.3.x: pytorch/pytorch:VERSION-cuda12.1-cudnn8-devel
- torch 2.4.x-2.5.x: nvidia/cuda:12.1.1-devel-ubuntu20.04
- torch 2.6.x+: nvidia/cuda:12.4.1-devel-ubuntu22.04

### Install vLLM with --no-deps:
```dockerfile
RUN pip install vllm==<VERSION> --no-deps
RUN pip install -c /opt/constraints.txt <DEPS_FROM_VLLM_REQUIREMENTS>
```

### HARDCODE commit SHA in EXACTLY 3 places:
```dockerfile
# 1st: ENV
ENV SGLANG_COMMIT={commit_sha}

# 2nd: git checkout
RUN git clone https://github.com/sgl-project/sglang.git && \\
    cd sglang && git checkout {commit_sha}

# 3rd: verification
RUN cd /sgl-workspace/sglang && \\
    ACTUAL=$(git rev-parse HEAD) && \\
    EXPECTED="{commit_sha}" && \\
    test "$ACTUAL" = "$EXPECTED" || exit 1 && \\
    echo "$ACTUAL" > /opt/sglang_commit.txt
```

### Install SGLang with --no-deps:
```dockerfile
RUN pip install -e /sgl-workspace/sglang/python --no-deps
RUN pip install -c /opt/constraints.txt <DEPS_FROM_PYPROJECT>
```

### Build settings:
- TORCH_CUDA_ARCH_LIST="9.0" (H100)
- MAX_JOBS=96

---

## FULL COMMIT SHA:
{commit_sha}

## pyproject.toml at this commit:
```toml
{pyproject_content}
```

## Current Dockerfile (reference):
```dockerfile
{dockerfile_content}
```

---

## OUTPUT

Write to: fixed-dockerfiles/{pr_date}/{short_hash}.Dockerfile

---

## VERIFICATION (DO ALL THREE)

1. Read back the file you wrote
2. Count SHA occurrences (must be EXACTLY 3)
3. For EACH pinned version, confirm you discovered it via WebFetch/WebSearch

---

## COMMON PITFALLS TO AVOID

- outlines 0.0.39+ requires pydantic v2, NOT v1
- If you pin pydantic v1 but outlines needs v2, pip will fail
- typing_extensions >= 4.14 has Sentinel (too new for old pydantic-core)
- Use typing_extensions 4.11.0 for May 2024 era builds
- pydantic 2.7.1 (April 2024) works with typing_extensions 4.11.0

Think step by step. WebFetch/WebSearch FIRST. Use discovered versions. Verify everything.
'''


def fix_dockerfile_with_claude(commit_sha: str, pr_date: str, index: int, total: int, timeout: int) -> tuple[str, bool, str]:
    """Use Claude Code CLI to fix a single Dockerfile. Returns (commit_sha, success, message)."""
    short_hash = commit_sha[:8]

    try:
        # Get pyproject.toml
        pyproject = get_pyproject_toml(commit_sha)

        # Get current Dockerfile
        dockerfile_path = get_dockerfile_path(commit_sha, pr_date)
        if not dockerfile_path.exists():
            return (commit_sha, False, f"No Dockerfile at {dockerfile_path}")

        dockerfile_content = dockerfile_path.read_text()

        # Build prompt
        prompt = build_claude_prompt(commit_sha, pr_date, pyproject, dockerfile_content)

        # Create output directory
        output_dir = FIXED_DIR / pr_date
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{pr_date}_{short_hash}.log"

        safe_print(f"[{index}/{total}] Starting {pr_date}/{short_hash}...")

        # Invoke Claude Code CLI with extended thinking (ultrathink)
        # Using --model opus for best reasoning
        # Using --betas for extended thinking capability
        result = subprocess.run(
            [
                "claude",
                "-p", prompt,
                "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep,WebFetch,WebSearch",
                "--model", "opus",  # Use opus for best reasoning
                "--betas", "interleaved-thinking-2025-05-14",  # Enable extended thinking
                "--verbose",
                "--dangerously-skip-permissions",  # Auto-approve for batch processing
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ,
        )

        # Log output
        with open(log_file, "w") as f:
            f.write(f"=== COMMIT ===\n{commit_sha}\n\n")
            f.write(f"=== DATE ===\n{pr_date}\n\n")
            f.write(f"=== PROMPT ===\n{prompt}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
            f.write(f"=== RETURN CODE ===\n{result.returncode}\n")

        # Check if output file was created
        output_file = output_dir / f"{short_hash}.Dockerfile"
        if output_file.exists():
            # Verify the commit SHA appears EXACTLY 3 times in the output
            content = output_file.read_text()
            sha_count = content.count(commit_sha)
            if sha_count >= 3:
                safe_print(f"[{index}/{total}] ✓ {pr_date}/{short_hash} - SUCCESS (SHA appears {sha_count}x)")
                return (commit_sha, True, f"Success - SHA appears {sha_count} times")
            elif sha_count > 0:
                safe_print(f"[{index}/{total}] ⚠ {pr_date}/{short_hash} - WARNING: SHA only appears {sha_count}x (need 3)")
                return (commit_sha, False, f"SHA only appears {sha_count} times, need at least 3")
            else:
                safe_print(f"[{index}/{total}] ✗ {pr_date}/{short_hash} - FAILED: SHA not found in output")
                return (commit_sha, False, "SHA not found in output Dockerfile")
        else:
            safe_print(f"[{index}/{total}] ✗ {pr_date}/{short_hash} - FAILED (no output file)")
            return (commit_sha, False, "No output file created")

    except subprocess.TimeoutExpired:
        safe_print(f"[{index}/{total}] ✗ {pr_date}/{short_hash} - TIMEOUT")
        return (commit_sha, False, "Timeout")
    except Exception as e:
        safe_print(f"[{index}/{total}] ✗ {pr_date}/{short_hash} - ERROR: {e}")
        return (commit_sha, False, str(e))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix SGLang Dockerfiles using Claude Code CLI with extended thinking"
    )
    parser.add_argument(
        "--parallel", "-j", type=int, default=DEFAULT_PARALLEL,
        help=f"Number of parallel Claude instances (default: {DEFAULT_PARALLEL})"
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per Dockerfile (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running Claude"
    )
    parser.add_argument(
        "--commit", type=str,
        help="Process only this commit (prefix match)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip commits that already have fixed Dockerfiles"
    )
    return parser.parse_args()


def main():
    """Process all commits from JSONL in parallel."""
    args = parse_args()

    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found")
        sys.exit(1)

    if not SGLANG_REPO.exists():
        print(f"ERROR: {SGLANG_REPO} not found")
        print("Run: git clone --bare https://github.com/sgl-project/sglang.git sglang.git")
        sys.exit(1)

    if not GUIDELINES_PATH.exists():
        print(f"ERROR: {GUIDELINES_PATH} not found")
        print("Copy the guidelines file first")
        sys.exit(1)

    # Load commits
    commits = []
    with open(JSONL_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                commits.append(obj)

    print(f"Loaded {len(commits)} commit entries from {JSONL_PATH}")

    # Collect all unique commits (commit_hash + parent_commit)
    all_commits = []
    for obj in commits:
        all_commits.append((obj["commit_hash"], obj["pr_date"]))
        all_commits.append((obj["parent_commit"], obj["pr_date"]))

    # Deduplicate while preserving order
    seen = set()
    unique_commits = []
    for sha, date in all_commits:
        if sha not in seen:
            seen.add(sha)
            unique_commits.append((sha, date))

    # Filter by specific commit if requested
    if args.commit:
        unique_commits = [(sha, date) for sha, date in unique_commits if sha.startswith(args.commit)]
        if not unique_commits:
            print(f"ERROR: No commit found matching '{args.commit}'")
            sys.exit(1)

    # Skip existing if requested
    if args.skip_existing:
        filtered = []
        for sha, date in unique_commits:
            output_file = FIXED_DIR / date / f"{sha[:8]}.Dockerfile"
            if not output_file.exists():
                filtered.append((sha, date))
            else:
                print(f"Skipping {sha[:8]} (already exists)")
        unique_commits = filtered

    total = len(unique_commits)
    print(f"Processing {total} unique commits")
    print(f"Running {args.parallel} Claude instances in parallel")
    print(f"Timeout: {args.timeout}s per Dockerfile")
    print(f"Model: opus with extended thinking")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN - Would process:")
        for sha, date in unique_commits:
            print(f"  {date}/{sha[:8]}")
        return

    # Process in parallel
    results = {"success": [], "failed": [], "skipped": []}

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {
            executor.submit(fix_dockerfile_with_claude, sha, date, i+1, total, args.timeout): (sha, date)
            for i, (sha, date) in enumerate(unique_commits)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            sha, date = futures[future]
            try:
                commit_sha, success, message = future.result()
                if success:
                    results["success"].append((commit_sha, message))
                elif "No Dockerfile" in message:
                    results["skipped"].append((commit_sha, message))
                else:
                    results["failed"].append((commit_sha, message))
            except Exception as e:
                results["failed"].append((sha, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Success: {len(results['success'])}")
    print(f"✗ Failed:  {len(results['failed'])}")
    print(f"⊘ Skipped: {len(results['skipped'])}")

    if results["failed"]:
        print("\nFailed commits:")
        for sha, msg in results["failed"]:
            print(f"  {sha[:8]}: {msg}")

    # Write summary to file
    summary_file = LOG_DIR / "summary.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "total": total,
            "success": len(results["success"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
            "success_commits": [s[0] for s in results["success"]],
            "failed_commits": [(s[0], s[1]) for s in results["failed"]],
            "skipped_commits": [(s[0], s[1]) for s in results["skipped"]],
        }, f, indent=2)
    print(f"\nDetailed summary written to: {summary_file}")


if __name__ == "__main__":
    main()
