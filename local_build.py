#!/usr/bin/env python3
"""
Local builder for commit-tagged Docker images (vLLM, SGLang).
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
    from tqdm import tqdm as _tqdm
    def _progress(total: int, desc: str):
        return _tqdm(total=total, desc=desc)
except Exception:
    def _progress(total: int, desc: str):
        class _Dummy:
            def update(self, n: int): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
        return _Dummy()

DEFAULT_IMAGE_NAME = "nvidia-vllm-docker"
DEFAULT_REPO_URL = "https://github.com/vllm-project/vllm.git"

def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    process = subprocess.Popen(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def run_command_stream(command: List[str], cwd: Optional[Path], line_prefix: str, log_file: Optional[Path] = None) -> Tuple[int, str]:
    process = subprocess.Popen(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, universal_newlines=True,
    )
    tail = collections.deque(maxlen=200)
    log_fp = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_file.open("w", encoding="utf-8")
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        tail.append(line)
        formatted = f"{line_prefix} {line}"
        print(formatted)
        if log_fp: log_fp.write(formatted + "\n")
    rc = process.wait()
    if log_fp: log_fp.close()
    return rc, "\n".join(tail)

def read_blacklist(blacklist_path: Path) -> Set[str]:
    if not blacklist_path.exists(): return set()
    return {line.strip() for line in blacklist_path.read_text().splitlines() if line.strip()}

def fetch_existing_tags(dockerhub_username: str, image_name: str) -> Set[str]:
    import urllib.request
    tags: Set[str] = set()
    repo = f"{dockerhub_username}/{image_name}"
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"
    while url and url != "null":
        try:
            with urllib.request.urlopen(url) as resp:
                payload = json.loads(resp.read())
            for item in payload.get("results", []):
                name = item.get("name")
                if name: tags.add(name)
            url = payload.get("next")
        except: break
    return tags

def iter_commits_with_dockerfile(dataset_path: Path) -> Iterable[Tuple[str, Optional[str]]]:
    with dataset_path.open("r") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            commit = obj.get("commit")
            if commit: yield commit, obj.get("Dockerfile")

def ensure_repo_cloned(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists(): return
    rc, out, err = run_command(["git", "clone", repo_url, str(repo_dir)])
    if rc != 0: raise RuntimeError(f"git clone failed: {err or out}")

_repo_lock = threading.Lock()

def _materialize_commit_tree(repo_dir: Path, commit_sha: str, dest_dir: Path) -> Tuple[bool, str]:
    with _repo_lock:
        run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
    if dest_dir.exists(): shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = (dest_dir.parent / f"{dest_dir.name}.tar").resolve()
    rc, _, err = run_command(["git", "archive", "--format=tar", "-o", str(tar_path), commit_sha], cwd=repo_dir)
    if rc != 0: return False, err
    rc2, _, err2 = run_command(["tar", "-C", str(dest_dir), "-xf", str(tar_path)])
    tar_path.unlink(missing_ok=True)
    return (rc2 == 0), (err2 if rc2 != 0 else "ok")

def resolve_dockerfile(repo_dir: Path) -> Optional[Path]:
    candidates = [repo_dir / "docker" / "Dockerfile", repo_dir / "Dockerfile", repo_dir / "examples" / "usage" / "triton" / "Dockerfile"]
    for path in candidates:
        if path.exists(): return path
    return None

def _apply_generic_dockerfile_fixes(dockerfile_path: Path) -> None:
    try:
        df_text = dockerfile_path.read_text()
        new_text = df_text

        # Detect focal (Ubuntu 20.04)
        is_focal = "ubuntu20.04" in new_text.lower() or "ubuntu:20.04" in new_text.lower() or "focal" in new_text.lower()

        # 1) Uppercase AS
        new_text = re.sub(r"(?im)^(\s*FROM\b[^\n]*?)\s+as\s+(\w+)", lambda m: f"{m.group(1)} AS {m.group(2)}", new_text)

        # 2) Normalize ENV
        def _normalize_env(line: str) -> str:
            if not re.match(r"^\s*ENV\b", line): return line
            after = re.sub(r"^\s*ENV\s+", "", line)
            if "=" in after: return line
            parts = after.strip().split(None, 1)
            if len(parts) == 2: return f"ENV {parts[0]}={parts[1]}"
            return line
        new_text = "\n".join(_normalize_env(l) for l in new_text.splitlines())

        # 3) Remove .git bind mounts
        new_text = re.sub(r"\s*--mount=type=bind,\s*source=\.git,\s*target=\.git\\?\s*\\?\\?", "", new_text)

        # 4) Focal / Python 3.10 Fix - Simplified and robust
        if is_focal:
            # Inject Miniforge at the top (no ToS requirements unlike Miniconda)
            conda_install = textwrap.dedent("""
                RUN apt-get update && apt-get install -y wget bzip2 ca-certificates && \\
                    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \\
                    bash miniforge.sh -b -p /opt/conda && rm miniforge.sh && \\
                    /opt/conda/bin/conda install -y python=3.10 && \\
                    ln -sf /opt/conda/bin/python3 /usr/bin/python3 && \\
                    ln -sf /opt/conda/bin/python3 /usr/bin/python3.10 && \\
                    ln -sf /opt/conda/bin/pip /usr/bin/pip && \\
                    ln -sf /opt/conda/bin/pip /usr/bin/pip3
                ENV PATH=/opt/conda/bin:$PATH
            """).strip()
            new_text = re.sub(r"(^FROM\s+.*?\n)", "\\1" + conda_install + "\n", new_text, count=1, flags=re.MULTILINE)
            
            # Remove any add-apt-repository deadsnakes lines  
            new_text = re.sub(r"RUN\s+.*?add-apt-repository\s+.*?ppa:deadsnakes/ppa.*?\n", "RUN echo 'Skipping deadsnakes PPA'\n", new_text)
            
            # Patch out python3.10 installs from apt (order matters: specific packages first!)
            # Use negative lookahead to avoid partial matches
            new_text = re.sub(r"python3\.10-distutils", "", new_text)
            new_text = re.sub(r"python3\.10-venv", "", new_text)
            new_text = re.sub(r"python3\.10-dev", "", new_text)
            # Only match standalone python3.10, not part of python3.10-something
            new_text = re.sub(r"python3\.10(?!-)", "", new_text)
        else:
            # For non-focal, still harden PPA
            ppa_fix = (
                "(DEBIAN_FRONTEND=noninteractive apt-get update && "
                "apt-get install -y gnupg2 ca-certificates curl lsb-release && "
                "CODENAME=$(lsb_release -sc 2>/dev/null || . /etc/os-release && echo $VERSION_CODENAME); "
                "for i in 1 2 3; do apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA6932366A755776 && break || sleep 2; done && "
                "echo \"deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu $CODENAME main\" > /etc/apt/sources.list.d/deadsnakes.list && "
                "apt-get update)"
            )
            new_text = re.sub(r"add-apt-repository\s+(?:-y\s+)?ppa:deadsnakes/ppa(?:\s+-y)?", ppa_fix, new_text)

        # 5) Fix stray RUN
        lines_fix = []
        lines_src = new_text.splitlines()
        i = 0
        while i < len(lines_src):
            cur = lines_src[i]
            if cur.strip().upper() == "RUN" and i+1 < len(lines_src):
                nxt = lines_src[i+1]
                if not re.match(r"^\s*(FROM|RUN|CMD|LABEL|MAINTAINER|EXPOSE|ENV|ADD|COPY|ENTRYPOINT|VOLUME|USER|WORKDIR|ARG|ONBUILD|STOPSIGNAL|HEALTHCHECK|SHELL)\b", nxt.strip(), re.IGNORECASE):
                    lines_fix.append("RUN " + nxt.lstrip())
                    i += 2; continue
            lines_fix.append(cur); i += 1
        new_text = "\n".join(lines_fix)

        # 6) Compilation and DeepEP/nvshmem fixes
        if "DeepEP" in new_text:
            # Ensure torch is present before cloning/building DeepEP
            new_text = new_text.replace(
                "git clone https://github.com/deepseek-ai/DeepEP.git",
                "python3 -m pip install torch && git clone https://github.com/deepseek-ai/DeepEP.git"
            )
            # Disable build isolation to use the 'torch' we just installed
            new_text = new_text.replace("pip install .", "python3 -m pip install --no-build-isolation .")
            new_text = new_text.replace("pip3 install .", "python3 -m pip install --no-build-isolation .")
            # Limit jobs to prevent OOM
            new_text = new_text.replace("python3 -m pip install --no-build-isolation .", "MAX_JOBS=4 python3 -m pip install --no-build-isolation .")

        new_text = new_text.replace("git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch", "git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch || true")
        new_text = new_text.replace("sed -i '1i#include <unistd.h>' examples/moe_shuffle.cu", "find . -name \"*.cu\" -o -name \"*.cpp\" -o -name \"*.h\" | xargs sed -i '1i#include <unistd.h>' || true")
        new_text = new_text.replace("cmake --build build --target install -j", "cmake --build build --target install -j4")

        if new_text != df_text: dockerfile_path.write_text(new_text)
    except Exception as e: print(f"Fix error: {e}")

def _apply_context_fixes(worktree_dir: Path, dockerfile_path: Optional[Path], project: str = "vllm") -> None:
    if project != "vllm":
        if dockerfile_path and dockerfile_path.exists(): _apply_generic_dockerfile_fixes(dockerfile_path)
        return
    if dockerfile_path and dockerfile_path.exists():
        txt = dockerfile_path.read_text()
        txt = txt.replace("outlines", "outlines<0.0.43")
        dockerfile_path.write_text(txt)

def build_one_commit(repo_dir: Path, commit_sha: str, image_name: str, dockerhub_username: str, platform: str, push: bool, cache_from: str|None, cache_to: str|None, dataset_dockerfile: str|None, show_build_logs: bool, project: str) -> Tuple[str, bool, str]:
    repo_dir_abs = repo_dir.resolve()
    worktree_dir = repo_dir_abs.parent / ".build-contexts" / commit_sha
    ok_ctx, msg = _materialize_commit_tree(repo_dir, commit_sha, worktree_dir)
    if not ok_ctx: return commit_sha, False, f"Context fail: {msg}"
    try:
        dockerfile_path = resolve_dockerfile(worktree_dir)
        if not dockerfile_path and project == "sglang":
            dockerfile_path = worktree_dir / "Dockerfile"
            dockerfile_path.write_text("FROM nvcr.io/nvidia/tritonserver:24.04-py3-min\nENV DEBIAN_FRONTEND=noninteractive\nRUN apt update && apt install -y python3 python3-pip curl git sudo\nWORKDIR /sgl-workspace\nRUN git clone --depth=1 https://github.com/sgl-project/sglang.git && cd sglang && pip install -e \"python[all]\"")
        if not dockerfile_path: return commit_sha, False, "No Dockerfile"
        _apply_context_fixes(worktree_dir, dockerfile_path, project)
        tag = f"docker.io/{dockerhub_username}/{image_name}:{commit_sha}"
        cmd = ["docker", "buildx", "build", "--platform", platform, "--file", str(dockerfile_path), "--tag", tag, "--label", f"org.opencontainers.image.revision={commit_sha}"]
        if show_build_logs: cmd += ["--progress", "plain"]
        if cache_from: cmd += ["--cache-from", cache_from]
        if cache_to: cmd += ["--cache-to", cache_to]
        if push: cmd.append("--push")
        cmd.append(str(worktree_dir))
        
        status_dir = repo_dir_abs.parent / ".build-status"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / f"{commit_sha}.building").write_text("building")
        
        if show_build_logs:
            rc, tail = run_command_stream(cmd, None, f"[{commit_sha[:7]}]", repo_dir_abs.parent / ".build-logs" / f"{commit_sha}.log")
            ok, msg = (rc == 0), ("ok" if rc == 0 else tail)
        else:
            rc, out, err = run_command(cmd)
            ok, msg = (rc == 0), ("ok" if rc == 0 else err or out)
            if not ok:
                log_dir = repo_dir_abs.parent / ".build-logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / f"{commit_sha}.log").write_text(out + "\n" + err)
        
        (status_dir / f"{commit_sha}.building").unlink(missing_ok=True)
        (status_dir / f"{commit_sha}.done").write_text("ok" if ok else "fail")
        return commit_sha, ok, msg
    finally:
        shutil.rmtree(worktree_dir, ignore_errors=True)

def _active_buildx_driver() -> Optional[str]:
    rc, out, _ = run_command(["docker", "buildx", "ls"])
    if rc != 0: return None
    for line in out.splitlines():
        if "*" in line and "NAME/" not in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) > 1: return parts[1].split()[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockerhub-username", required=True)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--dataset", default="nvidia-vllm-docker.jsonl")
    parser.add_argument("--blacklist", default="blacklist.txt")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--workdir", default="vllm")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-pushed", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--cache-dir", default=".buildx-cache")
    parser.add_argument("--show-build-logs", action="store_true")
    parser.add_argument("--project", default="vllm", choices=["vllm", "sglang"])
    args = parser.parse_args()

    dataset_path, blacklist_path, repo_dir = Path(args.dataset), Path(args.blacklist), Path(args.workdir)
    ensure_repo_cloned(repo_dir, args.repo_url)
    blacklist = read_blacklist(blacklist_path)
    records = list(iter_commits_with_dockerfile(dataset_path))
    filtered = [(c, d) for (c, d) in records if c not in blacklist]
    if args.skip_pushed:
        existing = fetch_existing_tags(args.dockerhub_username, args.image_name)
        filtered = [(c, d) for (c, d) in filtered if c not in existing]
    
    to_build = filtered[:args.batch_size]
    if not to_build: print("Nothing to build"); return 0
    print(f"Building {len(to_build)} commits (parallel={args.max_parallel})...")

    driver = _active_buildx_driver() or ""
    cache_from, cache_to = None, None
    if driver != "docker":
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_from = f"type=local,src={cache_dir}"
        cache_to = f"type=local,dest={cache_dir},mode=max"

    def _worker(item):
        return build_one_commit(repo_dir, item[0], args.image_name, args.dockerhub_username, args.platform, not args.no_push, cache_from, cache_to, item[1], args.show_build_logs, args.project)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(_worker, item): item[0] for item in to_build}
        with _progress(total=len(to_build), desc="Building") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                pbar.update(1)
                print(f"{res[0]} => {'OK' if res[1] else 'FAIL'}")

    return 1 if any(not r[1] for r in results) else 0

if __name__ == "__main__":
    raise SystemExit(main())
