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
python3 local_build.py --dockerhub-username DFUSERNAME --batch-size 1 --max-parallel 1 --skip-pushed --platform linux/amd64 --show-build-logs
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
import threading
import re
import shutil
import collections

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


def run_command_stream(
    command: List[str],
    cwd: Optional[Path],
    line_prefix: str,
) -> Tuple[int, str]:
    """Run a command and stream stdout/stderr merged line-by-line with a prefix.

    Returns (return_code, tail_output) where tail_output is the last ~200 lines
    concatenated, useful for error reporting without storing full logs.
    """
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    tail = collections.deque(maxlen=200)
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        tail.append(line)
        print(f"{line_prefix} {line}")
    rc = process.wait()
    return rc, "\n".join(tail)


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


_repo_lock = threading.Lock()


def _materialize_commit_tree(repo_dir: Path, commit_sha: str, dest_dir: Path) -> Tuple[bool, str]:
    """Export the repository at commit_sha into dest_dir using `git archive`.

    This avoids worktree checkout races and provides an isolated build context.
    """
    with _repo_lock:
        run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
    # Clean destination and recreate
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Create a tar archive and extract it
    tar_path = (dest_dir.parent / f"{dest_dir.name}.tar").resolve()
    rc, out, err = run_command(["git", "archive", "--format=tar", "-o", str(tar_path), commit_sha], cwd=repo_dir)
    if rc != 0:
        return False, (err or out)
    rc2, out2, err2 = run_command(["tar", "-C", str(dest_dir), "-xf", str(tar_path)])
    # Remove the temp tar regardless of success
    try:
        tar_path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    if rc2 != 0:
        return False, (err2 or out2)
    return True, "ok"


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
    dataset_dockerfile: Optional[str] = None,
    show_build_logs: bool = False,
) -> Tuple[str, bool, str]:
    repo_dir_abs = repo_dir.resolve()
    contexts_root = repo_dir_abs.parent / ".build-contexts"
    contexts_root.mkdir(parents=True, exist_ok=True)
    worktree_dir = contexts_root / commit_sha

    ok_ctx, ctx_msg = _materialize_commit_tree(repo_dir, commit_sha, worktree_dir)
    if not ok_ctx:
        return commit_sha, False, f"prepare context failed: {ctx_msg}"

    try:
        dockerfile_path = None
        if dataset_dockerfile:
            # Prefer Dockerfile content from dataset when provided
            dockerfile_path = worktree_dir / "Dockerfile"
            dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
            dockerfile_path.write_text(dataset_dockerfile)
        else:
            dockerfile_path = resolve_dockerfile(worktree_dir)
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
        if show_build_logs:
            command += ["--progress", "plain"]
        if cache_from:
            command += ["--cache-from", cache_from]
        if cache_to:
            command += ["--cache-to", cache_to]
        if push:
            command.append("--push")
        command.append(str(worktree_dir))

        # status markers
        repo_dir_abs = repo_dir.resolve()
        status_dir = repo_dir_abs.parent / ".build-status"
        status_dir.mkdir(parents=True, exist_ok=True)
        building_marker = status_dir / f"{commit_sha}.building"
        done_marker = status_dir / f"{commit_sha}.done"
        try:
            building_marker.write_text("building")
        except Exception:
            pass

        if show_build_logs:
            rc, tail = run_command_stream(command, cwd=None, line_prefix=f"[{commit_sha[:7]}]")
            ok = rc == 0
            msg = "ok" if ok else tail
        else:
            rc, out, err = run_command(command)
            ok = rc == 0
            msg = "ok" if ok else (err or out)
        try:
            building_marker.unlink(missing_ok=True)  # type: ignore[arg-type]
            done_marker.write_text("ok" if ok else "fail")
        except Exception:
            pass
        return commit_sha, ok, msg
    finally:
        # Clean up the context directory to free space
        try:
            shutil.rmtree(worktree_dir, ignore_errors=True)
        except Exception:
            pass


def _active_buildx_driver() -> Optional[str]:
    rc, out, err = run_command(["docker", "buildx", "ls"])
    if rc != 0:
        return None
    active_line: Optional[str] = None
    for line in out.splitlines():
        if "*" in line and not line.strip().startswith("NAME/"):
            active_line = line
            break
    if not active_line:
        return None
    parts = re.split(r"\s{2,}", active_line.strip())
    if len(parts) < 2:
        return None
    driver_field = parts[1]
    driver = driver_field.split()[0]
    return driver or None


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
    parser.add_argument(
        "--show-build-logs",
        action="store_true",
        help="Stream build logs (adds --progress=plain and prefixes each line)",
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

    records = list(iter_commits_with_dockerfile(dataset_path))
    filtered = [(c, d) for (c, d) in records if c not in blacklist]

    if args.skip_pushed:
        existing = fetch_existing_tags(args.dockerhub_username, args.image_name)
        filtered = [(c, d) for (c, d) in filtered if c not in existing]

    to_build = filtered[: args.batch_size]
    if not to_build:
        print("Nothing to build (after filtering).", file=sys.stderr)
        return 0

    print(f"Building {len(to_build)} commits (parallel={args.max_parallel})...")

    results: List[Tuple[str, bool, str]] = []
    # Enable local directory cache for shared layers across builds when supported
    driver = _active_buildx_driver() or ""
    cache_from: Optional[str]
    cache_to: Optional[str]
    if driver == "docker":
        # docker driver does not support cache export; disable cache flags
        cache_from = None
        cache_to = None
    else:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_from = f"type=local,src={cache_dir}"
        cache_to = f"type=local,dest={cache_dir},mode=max"

    def _worker(item: Tuple[str, Optional[str]]) -> Tuple[str, bool, str]:
        commit, dockerfile_text = item
        return build_one_commit(
            repo_dir=repo_dir,
            commit_sha=commit,
            image_name=args.image_name,
            dockerhub_username=args.dockerhub_username,
            platform=args.platform,
            push=not args.no_push,
            cache_from=cache_from,
            cache_to=cache_to,
            dataset_dockerfile=dockerfile_text,
            show_build_logs=args.show_build_logs,
        )

    if args.max_parallel <= 1:
        with _progress(total=len(to_build), desc="Building") as pbar:
            for item in to_build:
                res = _worker(item)
                results.append(res)
                pbar.update(1)
                print(f"{res[0]} => {'OK' if res[1] else 'FAIL'}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            future_to_commit = {ex.submit(_worker, item): item[0] for item in to_build}
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
            if len(to_build) == 1:
                # Show full error for single-commit troubleshooting
                print(f"- {commit}: {msg}", file=sys.stderr)
            else:
                summary = textwrap.shorten(msg.replace("\n", " "), width=200, placeholder="…")
                print(f"- {commit}: {summary}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


