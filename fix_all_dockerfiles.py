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
    return f'''You are fixing an SGLang Dockerfile. DISCOVER versions by exploring repos - NEVER guess.

## TASK: Fix Dockerfile for commit {commit_sha} (date: {pr_date})

## CORE PRINCIPLE: DISCOVER, Don't Guess

**NEVER use hardcoded version tables.** Always explore the actual repositories to discover compatible versions.

---

## STEP 0: EXPLORE THE ACTUAL REPOS (MANDATORY - DO THIS FIRST)

Before writing ANY Dockerfile, you MUST clone and explore repos to discover dependencies.

### 0a. Setup unique temp directory (parallel-safe)
```bash
WORK_DIR="/tmp/explore-{short_hash}"
rm -rf "$WORK_DIR" && mkdir -p "$WORK_DIR" && cd "$WORK_DIR"
```

### 0b. Clone and explore SGLang at this exact commit
```bash
git clone https://github.com/sgl-project/sglang.git sglang
cd sglang && git checkout {commit_sha}

# Find pyproject.toml location
find . -name "pyproject.toml" | head -5
cat python/pyproject.toml 2>/dev/null || cat pyproject.toml

# RECORD what you find:
# - torch version (in [project.optional-dependencies] srt section)
# - vllm version
# - flashinfer version
# - sgl-kernel version
# - transformers version
# - pydantic version requirements
```

### 0c. Clone vLLM at required version and read its ACTUAL requirements
```bash
cd "$WORK_DIR"
# Replace VERSION with what you found in SGLang's pyproject.toml
VLLM_VERSION="v0.4.2"  # CHANGE THIS based on pyproject.toml
git clone --depth 1 --branch $VLLM_VERSION https://github.com/vllm-project/vllm.git vllm

# Read vLLM's ACTUAL requirements - this is the source of truth
cat vllm/requirements-cuda.txt 2>/dev/null
cat vllm/requirements-common.txt 2>/dev/null
cat vllm/requirements.txt 2>/dev/null
cat vllm/pyproject.toml 2>/dev/null | head -100

# RECORD what vLLM actually needs:
# - torch version range
# - xformers version
# - pydantic version (v1.x or v2.x?)
# - other critical deps
```

### 0d. Check xformers compatibility (from vLLM requirements or releases)
```bash
# If vLLM requirements specify xformers, use that
# Otherwise clone xformers to check torch compatibility
git clone --depth 1 https://github.com/facebookresearch/xformers.git xformers
cat xformers/requirements.txt
```

### 0e. Read the guidelines file
```
Read the file: DOCKERFILE_FIX_GUIDELINES.md
```

### 0f. Cleanup after exploration
```bash
rm -rf "$WORK_DIR"
```

---

## FULL COMMIT SHA (HARDCODE this EXACT 40-char value in 3 places):
{commit_sha}

## pyproject.toml at this commit (for reference):
```toml
{pyproject_content}
```

## Current Dockerfile (reference only):
```dockerfile
{dockerfile_content}
```

---

## STEP 1: WEB SEARCHES TO VERIFY AVAILABILITY

After exploring repos, verify package availability:

### Check flashinfer wheels
WebFetch the URL matching your torch/CUDA (e.g., https://flashinfer.ai/whl/cu121/torch2.3/)
If empty/404 → BUILD FROM SOURCE

### Check sgl-kernel on PyPI
WebFetch https://pypi.org/simple/sgl-kernel/
If version not found → BUILD FROM SOURCE

### Check torch wheel exists
WebFetch https://download.pytorch.org/whl/cu121/ or cu124/

---

## STEP 2: WRITE DOCKERFILE USING DISCOVERED VERSIONS

### Base image (from torch version you discovered):
- torch 2.1.x-2.3.x: pytorch/pytorch:VERSION-cuda12.1-cudnn8-devel
- torch 2.4.x-2.5.x: nvidia/cuda:12.1.1-devel-ubuntu20.04
- torch 2.6.x+: nvidia/cuda:12.4.1-devel-ubuntu22.04

### CRITICAL: vLLM with --no-deps
```dockerfile
# WRONG - vLLM will pull wrong torch:
RUN pip install vllm==X.X.X

# CORRECT - use --no-deps, then install deps from your exploration:
RUN pip install vllm==X.X.X --no-deps
RUN pip install <deps-from-vllm-requirements.txt-you-discovered>
```

### Install deps from YOUR DISCOVERY (not hardcoded tables)
Use the versions you found in vLLM's requirements.txt:
- xformers version from vLLM requirements
- pydantic version from vLLM requirements (v1.x or v2.x)
- other deps from vLLM requirements

### HARDCODE commit SHA in EXACTLY 3 places:
```dockerfile
# 1st: ENV
ENV SGLANG_COMMIT={commit_sha}

# 2nd: git checkout
RUN git clone https://github.com/sgl-project/sglang.git sglang && \\
    cd sglang && git checkout {commit_sha}

# 3rd: verification
RUN cd /sgl-workspace/sglang && \\
    ACTUAL=$(git rev-parse HEAD) && \\
    EXPECTED="{commit_sha}" && \\
    test "$ACTUAL" = "$EXPECTED" || (echo "COMMIT MISMATCH" && exit 1) && \\
    echo "$ACTUAL" > /opt/sglang_commit.txt
```

### Patch pyproject.toml to remove pre-installed deps:
```dockerfile
RUN cd /sgl-workspace/sglang && \\
    sed -i 's/"flashinfer[^"]*",*//g' python/pyproject.toml && \\
    sed -i 's/"vllm[^"]*",*//g' python/pyproject.toml && \\
    sed -i 's/"sgl-kernel[^"]*",*//g' python/pyproject.toml
```

### Install SGLang (pyproject.toml is in python/ subdir):
```dockerfile
RUN pip install -e "python[all]"
```

### Build settings:
- TORCH_CUDA_ARCH_LIST="9.0" (H100)
- MAX_JOBS=96 (96-core machine)

---

## OUTPUT

Write to: fixed-dockerfiles/{pr_date}/{short_hash}.Dockerfile

---

## VERIFICATION

After writing:
1. Read back the file
2. Count full SHA occurrences (must be EXACTLY 3)
3. Verify all version pins came from your exploration, not guesses

---

## KEY REMINDERS

- DISCOVER versions by cloning repos and reading requirements
- vLLM ALWAYS with --no-deps
- Read vLLM's actual requirements.txt for xformers/pydantic versions
- pyproject.toml is in python/ subdirectory → use "python[all]"
- If pydantic conflict occurs, check what vLLM actually requires

Think step by step. Explore first. Use discovered versions. Verify everything.
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
