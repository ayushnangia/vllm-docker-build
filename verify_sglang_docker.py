#!/usr/bin/env python3
"""
Verify SGLang Docker images have correct commit-level installation.

Usage:
    python3 verify_sglang_docker.py <commit_sha>
    python3 verify_sglang_docker.py <commit_sha> --repo ayushnangia16/nvidia-sglang-docker
    python3 verify_sglang_docker.py --all  # verify all from success_with_dockerfile.txt
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path


def run_docker_cmd(image_tag: str, cmd: str) -> tuple[int, str]:
    """Run a command inside a docker container and return exit code + output."""
    full_cmd = f'docker run --rm {image_tag} bash -c "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    # Filter out Triton banner
    output = result.stdout
    lines = output.split('\n')
    filtered = []
    skip = False
    for line in lines:
        if '==================================' in line or 'Triton Inference Server' in line:
            skip = True
        elif 'https://docs.nvidia.com/datacenter/cloud-native/' in line:
            skip = False
            continue
        elif skip:
            continue
        else:
            filtered.append(line)
    return result.returncode, '\n'.join(filtered).strip()


def verify_commit(commit_sha: str, repo: str = "ayushnangia16/nvidia-sglang-docker") -> dict:
    """Verify a single commit's Docker image."""
    image_tag = f"{repo}:{commit_sha}"
    results = {
        "commit": commit_sha,
        "image": image_tag,
        "checks": {},
        "passed": True
    }

    # Check 1: Image exists locally or can be pulled
    print(f"Verifying {commit_sha[:12]}...")
    check_cmd = f"docker image inspect {image_tag}"
    result = subprocess.run(check_cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        # Try to check if it exists on DockerHub (without pulling)
        results["checks"]["image_exists"] = False
        results["passed"] = False
        print(f"  [FAIL] Image not found locally: {image_tag}")
        return results
    results["checks"]["image_exists"] = True

    # Check 2: Commit file exists and matches
    exit_code, output = run_docker_cmd(image_tag, "cat /opt/sglang_commit.txt")
    if exit_code != 0:
        results["checks"]["commit_file"] = {"exists": False}
        results["passed"] = False
        print(f"  [FAIL] /opt/sglang_commit.txt not found")
    else:
        commit_in_file = output.split('\n')[0].strip()
        matches = commit_in_file == commit_sha
        results["checks"]["commit_file"] = {
            "exists": True,
            "value": commit_in_file,
            "matches": matches
        }
        if not matches:
            results["passed"] = False
            print(f"  [FAIL] Commit mismatch: expected {commit_sha}, got {commit_in_file}")
        else:
            print(f"  [OK] Commit file: {commit_sha[:12]}")

    # Check 3: SGLang is installed
    exit_code, output = run_docker_cmd(image_tag, "pip show sglang 2>/dev/null | grep -E '^(Name|Version|Location|Editable)'")
    if exit_code != 0 or "Name: sglang" not in output:
        results["checks"]["sglang_installed"] = False
        results["passed"] = False
        print(f"  [FAIL] SGLang not installed")
    else:
        # Parse pip show output
        pip_info = {}
        for line in output.split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                pip_info[key.strip()] = val.strip()

        results["checks"]["sglang_installed"] = True
        results["checks"]["sglang_version"] = pip_info.get("Version", "unknown")
        results["checks"]["editable_install"] = "Editable" in output

        if "Editable" in output:
            print(f"  [OK] SGLang {pip_info.get('Version', '?')} (editable install)")
        else:
            print(f"  [WARN] SGLang {pip_info.get('Version', '?')} (NOT editable - may not be commit-specific)")

    # Check 4: SGLang can be imported
    # Run without filtering to check raw output for OK
    full_cmd = f'docker run --rm {image_tag} python3 -c "import sglang; print(\'SGLANG_IMPORT_OK\')"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or "SGLANG_IMPORT_OK" not in result.stdout:
        results["checks"]["sglang_import"] = False
        results["passed"] = False
        print(f"  [FAIL] SGLang import failed")
        if result.stderr:
            print(f"         Error: {result.stderr[:200]}")
    else:
        results["checks"]["sglang_import"] = True
        print(f"  [OK] SGLang import works")

    # Summary
    if results["passed"]:
        print(f"  [PASS] All checks passed for {commit_sha[:12]}")
    else:
        print(f"  [FAIL] Some checks failed for {commit_sha[:12]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify SGLang Docker images")
    parser.add_argument("commit", nargs="?", help="Commit SHA to verify")
    parser.add_argument("--repo", default="ayushnangia16/nvidia-sglang-docker", help="Docker repo")
    parser.add_argument("--all", action="store_true", help="Verify all commits from success_with_dockerfile.txt")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if args.all:
        # Read commits from success_with_dockerfile.csv (both human and parent commits)
        csv_file = Path(__file__).parent / "commit-status" / "success_with_dockerfile.csv"
        if not csv_file.exists():
            print(f"Error: {csv_file} not found")
            sys.exit(1)
        commits = []
        for line in csv_file.read_text().splitlines()[1:]:  # Skip header
            if line.strip() and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    commits.append(parts[0].strip())  # human commit
                    commits.append(parts[1].strip())  # parent commit
        commits = list(dict.fromkeys(commits))  # Remove duplicates, preserve order
    elif args.commit:
        commits = [args.commit]
    else:
        parser.print_help()
        sys.exit(1)

    all_results = []
    passed = 0
    failed = 0

    for commit in commits:
        result = verify_commit(commit, args.repo)
        all_results.append(result)
        if result["passed"]:
            passed += 1
        else:
            failed += 1
        print()

    # Summary
    print("=" * 50)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(commits)} commits")

    if args.json:
        print(json.dumps(all_results, indent=2))

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
