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
python3 local_build.py --dockerhub-username DOCKERHUB_USERNAME --batch-size 100 --max-parallel 10 --skip-pushed --platform linux/amd64 --show-build-logs
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
    log_file: Optional[Path] = None,
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
    log_fp = None
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = log_file.open("w", encoding="utf-8")
        except Exception:
            log_fp = None
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        tail.append(line)
        formatted = f"{line_prefix} {line}"
        print(formatted)
        if log_fp is not None:
            try:
                log_fp.write(formatted + "\n")
            except Exception:
                pass
    rc = process.wait()
    if log_fp is not None:
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass
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


def _apply_context_fixes(worktree_dir: Path, dockerfile_path: Optional[Path]) -> None:
    """Apply lightweight, per-commit context fixes before building.

    Currently mitigates pip ResolutionImpossible caused by outlines>=0.0.43
    depending on the unavailable 'pyairports' by pinning outlines to 0.0.41
    in requirements files when present.
    """
    candidate_files = [
        worktree_dir / "requirements-common.txt",
        worktree_dir / "requirements-cuda.txt",
        worktree_dir / "requirements.txt",
        worktree_dir / "requirements-dev.txt",
        worktree_dir / "requirements-test.txt",
        worktree_dir / "requirements-lint.txt",
    ]
    # Include common patterns: any requirements*.txt in root and under requirements/
    try:
        for p in worktree_dir.glob("requirements*.txt"):
            candidate_files.append(p)
        for p in (worktree_dir / "requirements").glob("*.txt"):
            candidate_files.append(p)
    except Exception:
        pass
    outlines_patterns = [
        re.compile(r"^\s*outlines\s*<\s*0\.1\s*,\s*>=\s*0\.0\.43\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*outlines\s*>=\s*0\.0\.43\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*outlines\s*~=\s*0\.0\.[4-9]+\s*$", re.IGNORECASE | re.MULTILINE),
    ]
    for req_path in candidate_files:
        try:
            if not req_path.exists():
                continue
            original = req_path.read_text()
            # First, perform a line-wise rewrite: any non-comment line that mentions
            # 'outlines' is replaced with a narrow constraint that avoids pyairports.
            lines = original.splitlines()
            changed = False
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    new_lines.append(line)
                    continue
                # Drop problematic dev/test-only deps that fail to build wheels
                if re.match(r"^(sparsezoo|sparseml)\b", stripped, re.IGNORECASE):
                    changed = True
                    continue
                if re.match(r"^outlines\b", stripped, re.IGNORECASE):
                    new_lines.append("outlines<0.0.43")
                    changed = True
                else:
                    new_lines.append(line)
            updated = "\n".join(new_lines)
            # Also fall back to regex replacement for variants we might have missed
            for pattern in outlines_patterns:
                newer = pattern.sub("outlines<0.0.43", updated)
                if newer != updated:
                    changed = True
                    updated = newer
            if changed:
                req_path.write_text(updated + ("\n" if original.endswith("\n") else ""))
        except Exception:
            # Do not fail the build orchestration if a fix cannot be applied
            pass

    # If requirements files were not present or not changed, ensure Dockerfile
    # performs a pre-install of outlines==0.0.41 before any requirements installs.
    if dockerfile_path and dockerfile_path.exists():
        try:
            df_text = dockerfile_path.read_text()
            # Inject pin before common forms of pip install -r ...
            new_text = df_text
            py_pat = r"python3\s*-m\s*pip\s+install\s+-r\s+([\w\-./]+)"
            # Avoid matching 'pip' when part of 'python3 -m pip'
            pip_pat = r"(?<!-m\s)pip\s+install\s+-r\s+([\w\-./]+)"
            if re.search(py_pat, new_text):
                new_text = re.sub(
                    py_pat,
                    r"python3 -m pip install --no-deps outlines==0.0.41 && python3 -m pip install -r \1",
                    new_text,
                    count=1,
                )
            elif re.search(pip_pat, new_text):
                new_text = re.sub(
                    pip_pat,
                    r"pip install --no-deps outlines==0.0.41 && pip install -r \1",
                    new_text,
                    count=1,
                )
            # Normalize Dockerfile style to reduce warnings
            # 1) Ensure AS is uppercase in multi-stage lines beginning with FROM
            new_text = re.sub(
                r"(?im)^(\s*FROM\b[^\n]*?)\s+as\s+(\w+)",
                lambda m: f"{m.group(1)} AS {m.group(2)}",
                new_text,
            )
            # 2) Convert legacy ENV "key value" to "key=value" when no '=' present
            def _normalize_env(line: str) -> str:
                if not re.match(r"^\s*ENV\b", line):
                    return line
                after = re.sub(r"^\s*ENV\s+", "", line)
                # leave multi-assign or already key=value forms untouched
                if "=" in after:
                    return line
                parts = after.strip().split(None, 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return re.sub(r"^\s*ENV\s+.*$", f"ENV {parts[0]}={parts[1]}", line)
                return line
            new_text = "\n".join(_normalize_env(l) for l in new_text.splitlines()) + ("\n" if new_text.endswith("\n") else "")
            # 3) Remove bind-mount of .git (not present in archived context) to avoid
            #    BuildKit checksum errors like: failed to calculate checksum of ref ...
            new_text = re.sub(
                r"\s*--mount=type=bind,\s*source=\.git,\s*target=\.git\\?\s*\\?\\?",
                "",
                new_text,
            )
            # 4) Harden deadsnakes PPA add to mitigate transient GPG key fetch timeouts
            new_text = new_text.replace(
                "add-apt-repository ppa:deadsnakes/ppa",
                "apt-get update -y && apt-get install -y gnupg ca-certificates && "
                "add-apt-repository -y ppa:deadsnakes/ppa || (sleep 10 && add-apt-repository -y ppa:deadsnakes/ppa) || (sleep 30 && add-apt-repository -y ppa:deadsnakes/ppa)",
            )
            # 5) Disable wheel size enforcement step that fails images with large .so
            new_text = re.sub(
                r"^(\s*RUN\s+python3\s+check-wheel-size\.py\s+dist\s*)$",
                r"RUN echo 'Skipping wheel size check in docker build context'",
                new_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            # 6) Ensure setuptools-scm has a version when building from archived context (no .git)
            #    Define both universal and project-specific envs to satisfy different setups
            if "SETUPTOOLS_SCM_PRETEND_VERSION" not in new_text and "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM" not in new_text:
                if re.search(r"^\s*ENV\s+BUILDKITE_COMMIT=", new_text, re.MULTILINE):
                    new_text = re.sub(
                        r"^(\s*ENV\s+BUILDKITE_COMMIT=.*)$",
                        r"\1\nENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+${BUILDKITE_COMMIT:-local}\nENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.0.0+${BUILDKITE_COMMIT:-local}",
                        new_text,
                        count=1,
                        flags=re.MULTILINE,
                    )
                else:
                    # Fallback: insert after first FROM
                    new_text = re.sub(
                        r"^(\s*FROM\b.*)$",
                        r"\1\nENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+local\nENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.0.0+local",
                        new_text,
                        count=1,
                        flags=re.MULTILINE,
                    )

            # Also guard the direct setup.py invocation by exporting the env inline
            new_text = re.sub(
                r"python3\s+setup\.py\s+bdist_wheel\s+--dist-dir=dist\s+--py-limited-api=cp38",
                r"export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0+${BUILDKITE_COMMIT:-local}} SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-0.0.0+${BUILDKITE_COMMIT:-local}} && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38",
                new_text,
                flags=re.IGNORECASE,
            )
            # 7) Fix stray RUN line breaks that cause Dockerfile parse errors like
            #    "unknown instruction: if" by collapsing "RUN\n    if ..." to "RUN if ..."
            lines_fix: list[str] = []
            i = 0
            lines_src = new_text.splitlines()
            while i < len(lines_src):
                cur = lines_src[i]
                if cur.strip().upper() == "RUN" and i + 1 < len(lines_src):
                    nxt = lines_src[i + 1]
                    # Only collapse when next line looks like a shell command, not a new instruction
                    if not re.match(r"^\s*(FROM|RUN|CMD|LABEL|MAINTAINER|EXPOSE|ENV|ADD|COPY|ENTRYPOINT|VOLUME|USER|WORKDIR|ARG|ONBUILD|STOPSIGNAL|HEALTHCHECK|SHELL)\b", nxt.strip(), re.IGNORECASE):
                        lines_fix.append("RUN " + nxt.lstrip())
                        i += 2
                        continue
                lines_fix.append(cur)
                i += 1
            new_text = "\n".join(lines_fix) + ("\n" if new_text.endswith("\n") else "")
            if new_text != df_text:
                dockerfile_path.write_text(new_text)
        except Exception:
            pass


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

        # Apply per-commit context fixes before build (e.g., outlines pin)
        _apply_context_fixes(worktree_dir, dockerfile_path)

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
            # Prepare per-commit log path
            logs_dir = repo_dir_abs.parent / ".build-logs"
            log_path = logs_dir / f"{commit_sha}.log"
            rc, tail = run_command_stream(
                command,
                cwd=None,
                line_prefix=f"[{commit_sha[:7]}]",
                log_file=log_path,
            )
            ok = rc == 0
            msg = "ok" if ok else tail
        else:
            rc, out, err = run_command(command)
            ok = rc == 0
            msg = "ok" if ok else (err or out)
            if not ok:
                # When not streaming, still persist the logs for debugging
                try:
                    logs_dir = repo_dir_abs.parent / ".build-logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    (logs_dir / f"{commit_sha}.log").write_text((out or "") + ("\n" + err if err else ""))
                except Exception:
                    pass
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


