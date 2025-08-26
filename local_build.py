#!/usr/bin/env python3
"""
Local builder for vLLM commit-tagged Docker images.

Features:
- Reads commits from nvidia-vllm-docker.jsonl (only those with a Dockerfile)
- Respects blacklist.txt
- Optionally skips commits that already have a tag on Docker Hub
- Batches and optional parallel builds
- Push optional (default: push enabled)

Requirements:
- Docker with Buildx enabled
- git

Example:
  python3 local_build.py \
    --dockerhub-username YOUR_NAME \
    --batch-size 20 \
    --max-parallel 2 \
    --skip-pushed
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

try:
    # Optional pretty progress bars; falls back silently if not installed
    from tqdm import tqdm as _tqdm

    def _progress(total: int, desc: str):
        return _tqdm(total=total, desc=desc)

except Exception:  # pragma: no cover
    def _progress(total: int, desc: str):  # type: ignore
        class _Dummy:
            def update(self, n: int):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Dummy()


DEFAULT_IMAGE_NAME = "nvidia-vllm-docker"
DEFAULT_REPO_URL = "https://github.com/vllm-project/vllm.git"


def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def read_blacklist(blacklist_path: Path) -> Set[str]:
    if not blacklist_path.exists():
        return set()
    commits: Set[str] = set()
    for line in blacklist_path.read_text().splitlines():
        sha = line.strip()
        if sha:
            commits.add(sha)
    return commits


def fetch_existing_tags(dockerhub_username: str, image_name: str) -> Set[str]:
    import urllib.request
    import urllib.error
    import urllib.parse

    tags: Set[str] = set()
    repo = f"{dockerhub_username}/{image_name}"
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"
    while url and url != "null":
        try:
            with urllib.request.urlopen(url) as resp:
                payload = json.loads(resp.read())
        except Exception:
            break
        for item in payload.get("results", []):
            name = item.get("name")
            if isinstance(name, str) and name:
                tags.add(name)
        url = payload.get("next")
    return tags


def iter_commits_with_dockerfile(dataset_path: Path) -> Iterable[Tuple[str, Optional[str]]]:
    with dataset_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            commit = obj.get("commit")
            dockerfile = obj.get("Dockerfile")
            if dockerfile is not None and isinstance(commit, str):
                yield commit, dockerfile


def ensure_repo_cloned(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists():
        return
    rc, out, err = run_command(["git", "clone", repo_url, str(repo_dir)])
    if rc != 0:
        raise RuntimeError(f"git clone failed: {err or out}")


def checkout_commit(repo_dir: Path, commit_sha: str) -> bool:
    run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
    rc, out, err = run_command(["git", "checkout", "-q", commit_sha], cwd=repo_dir)
    return rc == 0


def resolve_dockerfile(repo_dir: Path) -> Optional[Path]:
    candidates = [repo_dir / "docker" / "Dockerfile", repo_dir / "Dockerfile"]
    for path in candidates:
        if path.exists():
            return path
    return None


def build_one_commit(
    repo_dir: Path,
    commit_sha: str,
    image_name: str,
    dockerhub_username: str,
    platform: str,
    push: bool,
    cache_from: Optional[str] = None,
    cache_to: Optional[str] = None,
) -> Tuple[str, bool, str]:
    if not checkout_commit(repo_dir, commit_sha):
        return commit_sha, False, "checkout failed"
    dockerfile_path = resolve_dockerfile(repo_dir)
    if dockerfile_path is None:
        return commit_sha, False, "no Dockerfile"

    tag = f"docker.io/{dockerhub_username}/{image_name}:{commit_sha}"

    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        platform,
        "--file",
        str(dockerfile_path),
        "--tag",
        tag,
        "--label",
        f"org.opencontainers.image.revision={commit_sha}",
    ]
    if cache_from:
        command += ["--cache-from", cache_from]
    if cache_to:
        command += ["--cache-to", cache_to]
    if push:
        command.append("--push")
    command.append(str(repo_dir))

    rc, out, err = run_command(command)
    ok = rc == 0
    msg = "ok" if ok else (err or out)
    return commit_sha, ok, msg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local builder for vLLM commit-tagged Docker images",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dockerhub-username", required=True, help="Docker Hub username for tagging")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME, help="Image name (default: %(default)s)")
    parser.add_argument("--dataset", default="nvidia-vllm-docker.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--blacklist", default="blacklist.txt", help="Path to blacklist file")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="vLLM repo URL")
    parser.add_argument("--workdir", default="vllm", help="Where to clone the repo locally")
    parser.add_argument("--batch-size", type=int, default=10, help="How many commits to build in this run")
    parser.add_argument("--max-parallel", type=int, default=1, help="Parallel builds (be cautious locally)")
    parser.add_argument("--skip-pushed", action="store_true", help="Skip commits whose tags already exist")
    parser.add_argument("--no-push", action="store_true", help="Build without pushing")
    parser.add_argument("--platform", default="linux/amd64", help="Build platform (default: %(default)s)")
    parser.add_argument(
        "--cache-dir",
        default=".buildx-cache",
        help="Directory for local Buildx cache (shared layers across builds)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    dataset_path = Path(args.dataset)
    blacklist_path = Path(args.blacklist)
    repo_dir = Path(args.workdir)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    ensure_repo_cloned(repo_dir, args.repo_url)

    blacklist = read_blacklist(blacklist_path)

    all_commits = [c for c, _ in iter_commits_with_dockerfile(dataset_path)]
    filtered = [c for c in all_commits if c not in blacklist]

    if args.skip_pushed:
        existing = fetch_existing_tags(args.dockerhub_username, args.image_name)
        filtered = [c for c in filtered if c not in existing]

    to_build = filtered[: args.batch_size]
    if not to_build:
        print("Nothing to build (after filtering).", file=sys.stderr)
        return 0

    print(f"Building {len(to_build)} commits (parallel={args.max_parallel})...")

    results: List[Tuple[str, bool, str]] = []
    # Enable local directory cache for shared layers across builds
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_from = f"type=local,src={cache_dir}"
    cache_to = f"type=local,dest={cache_dir},mode=max"

    def _worker(commit: str) -> Tuple[str, bool, str]:
        return build_one_commit(
            repo_dir=repo_dir,
            commit_sha=commit,
            image_name=args.image_name,
            dockerhub_username=args.dockerhub_username,
            platform=args.platform,
            push=not args.no_push,
            cache_from=cache_from,
            cache_to=cache_to,
        )

    if args.max_parallel <= 1:
        with _progress(total=len(to_build), desc="Building") as pbar:
            for commit in to_build:
                res = _worker(commit)
                results.append(res)
                pbar.update(1)
                print(f"{res[0]} => {'OK' if res[1] else 'FAIL'}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            future_to_commit = {ex.submit(_worker, c): c for c in to_build}
            with _progress(total=len(to_build), desc="Building") as pbar:
                for fut in concurrent.futures.as_completed(future_to_commit):
                    res = fut.result()
                    results.append(res)
                    pbar.update(1)
                    print(f"{res[0]} => {'OK' if res[1] else 'FAIL'}")

    failures = [r for r in results if not r[1]]
    if failures:
        print("\nFailures:", file=sys.stderr)
        for commit, ok, msg in failures:
            summary = textwrap.shorten(msg.replace("\n", " "), width=200, placeholder="…")
            print(f"- {commit}: {summary}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


