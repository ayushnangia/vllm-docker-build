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

DEFAULT_IMAGE_NAME = "nvidia-sglang-docker"
DEFAULT_REPO_URL = "https://github.com/sgl-project/sglang.git"

# Commits that need flashinfer built from source (no prebuilt wheels for their torch version)
FLASHINFER_FROM_SOURCE_COMMITS = {
    "73b13e69", "8609e637", "880221bd", "8f3173d0",
}

# Commits that need sgl-kernel built from source (version on PyPI doesn't exist)
# Maps commit prefix -> (torch_version, cuda_index) for correct wheel compatibility
SGL_KERNEL_FROM_SOURCE_COMMITS = {
    "9c088829": ("2.6.0", "cu124"),   # sgl-kernel==0.0.9.post2
    "d1112d85": ("2.5.1", "cu124"),   # sgl-kernel==0.0.5.post2
    "93470a14": ("2.5.1", "cu124"),   # sgl-kernel==0.0.8
}

# Commits that need torchao version fix (torchao 0.9.0 incompatible with torch 2.7.1)
# torchao 0.12.0+ required for torch 2.7.1 per https://github.com/pytorch/ao/issues/2919
TORCHAO_FIX_COMMITS = {
    "a99801e0",  # has torchao==0.9.0 + torch==2.7.1 conflict
}

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

def _apply_generic_dockerfile_fixes(dockerfile_path: Path, commit_sha: str = "") -> None:
    try:
        df_text = dockerfile_path.read_text()
        new_text = df_text

        # === COMMIT PROOF INJECTION ===
        # Add ARG SGLANG_COMMIT if not present (will be passed from --build-arg)
        if "ARG SGLANG_COMMIT" not in new_text:
            # Insert after first FROM line
            new_text = re.sub(
                r'(FROM\s+[^\n]+\n)',
                r'\1ARG SGLANG_COMMIT\n',
                new_text,
                count=1
            )

        # Add commit proof file creation if not present
        if "/opt/sglang_commit.txt" not in new_text:
            # Find the last RUN command and add after it, or before final CMD/ENTRYPOINT
            commit_proof_cmd = '\n# Write commit proof for runtime verification\nRUN mkdir -p /opt && echo "${SGLANG_COMMIT:-unknown}" > /opt/sglang_commit.txt && echo "Patched by local_build.py" >> /opt/sglang_commit.txt\n'

            # Try to insert before CMD or ENTRYPOINT
            if re.search(r'\n(CMD|ENTRYPOINT)\s', new_text):
                new_text = re.sub(r'\n(CMD|ENTRYPOINT)\s', commit_proof_cmd + r'\n\1 ', new_text, count=1)
            else:
                # Append at end
                new_text = new_text.rstrip() + commit_proof_cmd

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

        # 4) Fix Python 3.10 on Ubuntu 20.04 (focal) - Build from source
        # The deadsnakes PPA no longer has packages for Ubuntu 20.04 (focal)
        # Solution: Build Python 3.10 from source instead
        if is_focal and "ppa:deadsnakes" in new_text:
            # Build Python 3.10 from source - this replaces the entire deadsnakes approach
            python_from_source = textwrap.dedent(r'''
                # Build Python 3.10 from source (deadsnakes PPA no longer supports focal)
                RUN apt-get update && apt-get install -y \
                    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
                    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
                    liblzma-dev tk-dev uuid-dev curl git sudo libibverbs-dev && \
                    cd /tmp && \
                    wget -q https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
                    tar -xzf Python-3.10.14.tgz && \
                    cd Python-3.10.14 && \
                    ./configure --enable-optimizations --enable-shared --with-ensurepip=install LDFLAGS="-Wl,-rpath /usr/local/lib" && \
                    make -j$(nproc) && \
                    make altinstall && \
                    ldconfig && \
                    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
                    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
                    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip && \
                    python3 --version && \
                    pip3 --version && \
                    cd / && rm -rf /tmp/Python-3.10.14* && \
                    rm -rf /var/lib/apt/lists/* && apt-get clean
            ''').strip()

            # Replace the entire RUN block that uses deadsnakes with our source build
            # Pattern: Match RUN block from tzdata echo to apt clean (handles backslash continuations)
            # Use re.DOTALL so .* matches newlines, and match non-greedily to the first "apt clean"
            deadsnakes_pattern = r"RUN\s+echo\s+'tzdata[^']+'\s*\|.*?&&\s*apt\s+clean"
            match = re.search(deadsnakes_pattern, new_text, re.DOTALL)
            if match:
                new_text = new_text[:match.start()] + python_from_source + new_text[match.end():]
            else:
                # Fallback: remove the entire RUN block containing deadsnakes and add Python from source
                # Try matching RUN ... deadsnakes ... ending with newline before next command
                fallback_pattern = r"RUN\s+[^\n]*deadsnakes[^\n]*(?:\\\n[^\n]*)*\n"
                if re.search(fallback_pattern, new_text):
                    new_text = re.sub(fallback_pattern, python_from_source + "\n", new_text)
                else:
                    # Last resort: Insert Python build after the first FROM line
                    new_text = re.sub(
                        r'(FROM\s+[^\n]+\n)',
                        r'\1\n' + python_from_source + '\n',
                        new_text,
                        count=1
                    )

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

        # 7) SGLang-specific fixes: Use build context instead of git clone or PyPI
        # This ensures we build the exact commit, not latest master or PyPI version

        # CRITICAL: Replace pip install "sglang[all]" from PyPI with local source install
        # This pattern installs whatever is on PyPI, not the archived commit!
        if 'pip' in new_text and '"sglang[all]"' in new_text and 'COPY' not in new_text:
            # Replace PyPI install with local source install
            new_text = re.sub(
                r'pip3?\s+(?:--no-cache-dir\s+)?install\s+"sglang\[all\]"',
                'pip3 --no-cache-dir install -e "/sgl-workspace/sglang/python[all]"',
                new_text
            )
            # Add COPY command before the install if not present
            if 'COPY . /sgl-workspace/sglang' not in new_text and 'COPY python' not in new_text:
                # Insert COPY after WORKDIR
                new_text = re.sub(
                    r'(WORKDIR\s+/sgl-workspace\s*\n)',
                    r'\1\n# Copy archived commit source (not PyPI)\nCOPY . /sgl-workspace/sglang\n',
                    new_text
                )

        if "sgl-project/sglang" in new_text:
            # The SGLang Dockerfile typically has a RUN command like:
            # RUN pip install ... && git clone ... && cd sglang && pip install -e "python[all]"
            # We need to split this into:
            # 1. RUN pip install ... (pre-clone setup)
            # 2. COPY . /sgl-workspace/sglang (use build context)
            # 3. RUN pip install -e "python[all]" (install sglang from local copy)

            # First, handle the complex case where git clone is inside a multi-command RUN
            # Split at "git clone" and reconstruct
            clone_pattern = r'(RUN\s+[^&]*(?:&&[^&]*)*?)(\s*&&\s*git\s+clone\s+[^\n]*sgl-project/sglang\.git[^\n]*)(\s*\\?\s*\n\s*&&\s*cd\s+sglang\s*\\?\s*\n)?(.*)'
            match = re.search(clone_pattern, new_text, re.DOTALL)
            if match:
                pre_clone = match.group(1).rstrip(' \\\n&')
                post_clone = match.group(4) if match.group(4) else ""
                # Clean up the post_clone part - remove leading && and cd sglang
                post_clone = re.sub(r'^\s*\\?\s*\n?\s*&&\s*cd\s+sglang\s*\\?\s*\n?\s*', '', post_clone)
                post_clone = re.sub(r'^\s*&&\s*', '', post_clone.strip())
                # Remove any remaining "cd sglang &&" at the start
                post_clone = re.sub(r'^cd\s+sglang\s*\\?\s*\n?\s*&&\s*', '', post_clone)
                # CRITICAL: Remove ALL "cd sglang" commands anywhere in post_clone
                # They appear as "&& cd sglang \\" or "&& cd sglang &&" in multi-line RUN
                post_clone = re.sub(r'&&\s*cd\s+sglang\s*\\?\s*\n?\s*&&', '&&', post_clone)
                post_clone = re.sub(r'&&\s*cd\s+sglang\s*\\?\s*\n?\s*', '', post_clone)
                post_clone = re.sub(r'\s*\\?\s*\n\s*&&\s*cd\s+sglang\s*', '', post_clone)
                # Clean up any double && that might result
                post_clone = re.sub(r'&&\s*&&', '&&', post_clone)

                # Reconstruct:
                # 1. Pre-clone RUN (if there's actual content beyond just "RUN")
                # 2. COPY for sglang (into python subdir since that's where pyproject.toml is)
                # 3. Post-clone RUN in sglang directory
                replacement_parts = []
                if pre_clone.strip() and pre_clone.strip() != "RUN":
                    replacement_parts.append(pre_clone + "\n")
                replacement_parts.append("\nCOPY . /sgl-workspace/sglang\n")
                if post_clone.strip():
                    # The post_clone likely has the pip install command
                    replacement_parts.append(f"RUN cd /sgl-workspace/sglang && {post_clone}")
                # Trust pyproject.toml for all dependencies - no extra pins needed
                # The commit's pyproject.toml has the correct versions for that point in time
                pass

                new_text = new_text[:match.start()] + "".join(replacement_parts) + new_text[match.end():]
            else:
                # Simple case: standalone git clone line
                new_text = re.sub(
                    r'RUN\s+git\s+clone\s+[^\n]*sgl-project/sglang\.git[^\n]*\n',
                    'COPY python /sgl-workspace/sglang/python\nCOPY docker /sgl-workspace/sglang/docker\n',
                    new_text
                )

            # Trust pyproject.toml for flashinfer/torch versions
            # Each commit's Dockerfile and pyproject.toml have the correct versions
            pass

        # Inject flashinfer build-from-source for commits without prebuilt wheels
        if commit_sha:
            for prefix in FLASHINFER_FROM_SOURCE_COMMITS:
                if commit_sha.startswith(prefix):
                    # Add flashinfer source build before sglang install
                    # Must use --no-build-isolation and pre-install deps to avoid PyPI wheel
                    # Also patch pyproject.toml to remove flashinfer dep (already installed from source)
                    flashinfer_build = '''# Build flashinfer from source (no prebuilt wheel for this torch version)
RUN pip install torch==2.7.1 ninja numpy --extra-index-url https://download.pytorch.org/whl/cu126 && \\
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \\
    cd /tmp/flashinfer && git checkout v0.2.6.post1 && \\
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MAX_JOBS=4 pip install --no-build-isolation --no-deps -v . && \\
    rm -rf /tmp/flashinfer

# Remove flashinfer from pyproject.toml deps (already built from source)
RUN sed -i 's/"flashinfer_python[^"]*",/# flashinfer built from source/g' /sgl-workspace/sglang/python/pyproject.toml

'''
                    # Insert before the RUN that installs sglang (after COPY . /sgl-workspace/sglang)
                    new_text = re.sub(
                        r'(COPY \. /sgl-workspace/sglang\n)(RUN cd /sgl-workspace/sglang)',
                        r'\1' + flashinfer_build + r'\2',
                        new_text,
                        count=1
                    )
                    break

        # Inject sgl-kernel build-from-source for commits where PyPI version doesn't exist
        if commit_sha:
            for prefix, (torch_ver, cuda_idx) in SGL_KERNEL_FROM_SOURCE_COMMITS.items():
                if commit_sha.startswith(prefix):
                    # Build sgl-kernel from the included source folder
                    # Must install torch first as sgl-kernel needs it for CMake
                    # Then patch pyproject.toml to remove the version constraint
                    sgl_kernel_build = f'''# Build sgl-kernel from source (required version not on PyPI)
RUN pip install torch=={torch_ver} --index-url https://download.pytorch.org/whl/{cuda_idx} && \\
    cd /sgl-workspace/sglang/sgl-kernel && \\
    pip install scikit-build-core ninja cmake && \\
    pip install --no-build-isolation -v .

# Remove sgl-kernel version constraint from pyproject.toml (already built from source)
RUN sed -i 's/"sgl-kernel==[^"]*"/"sgl-kernel"/g' /sgl-workspace/sglang/python/pyproject.toml

'''
                    # Insert before the RUN that installs sglang (after COPY . /sgl-workspace/sglang)
                    new_text = re.sub(
                        r'(COPY \. /sgl-workspace/sglang\n)(RUN cd /sgl-workspace/sglang)',
                        r'\1' + sgl_kernel_build + r'\2',
                        new_text,
                        count=1
                    )
                    break

        # Fix torchao version conflict (0.9.0 incompatible with torch 2.7.1)
        if commit_sha:
            for prefix in TORCHAO_FIX_COMMITS:
                if commit_sha.startswith(prefix):
                    # Patch pyproject.toml to use torchao>=0.12.0 instead of 0.9.0
                    torchao_fix = '''# Fix torchao version (0.9.0 incompatible with torch 2.7.1, need 0.12.0+)
RUN sed -i 's/"torchao==0.9.0"/"torchao>=0.12.0"/g' /sgl-workspace/sglang/python/pyproject.toml

'''
                    # Insert after COPY . /sgl-workspace/sglang
                    new_text = re.sub(
                        r'(COPY \. /sgl-workspace/sglang\n)',
                        r'\1' + torchao_fix,
                        new_text,
                        count=1
                    )
                    break

        if new_text != df_text: dockerfile_path.write_text(new_text)
    except Exception as e: print(f"Fix error: {e}")

def _apply_context_fixes(worktree_dir: Path, dockerfile_path: Optional[Path], project: str = "vllm", commit_sha: str = "") -> None:
    if project != "vllm":
        if dockerfile_path and dockerfile_path.exists(): _apply_generic_dockerfile_fixes(dockerfile_path, commit_sha)
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
            # CRITICAL: Use COPY from build context, NOT git clone (which would get HEAD)
            # The build context is already the archived commit via _materialize_commit_tree
            # Use official PyTorch CUDA image as base (NOT tritonserver)
            # See: https://hub.docker.com/r/pytorch/pytorch
            dockerfile_path.write_text(textwrap.dedent(f"""\
                # SGLang fallback Dockerfile - generated by local_build.py
                # Base: PyTorch with CUDA 12.1 (compatible with vllm>=0.4.2, flashinfer)
                FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

                ENV DEBIAN_FRONTEND=noninteractive
                ARG SGLANG_COMMIT={commit_sha}

                # Install system dependencies
                RUN apt-get update && apt-get install -y \\
                    git curl wget sudo \\
                    && rm -rf /var/lib/apt/lists/*

                WORKDIR /sgl-workspace

                # Copy archived commit contents (NOT git clone which would get HEAD)
                COPY . /sgl-workspace/sglang

                # Write commit proof file for runtime verification
                RUN mkdir -p /opt && echo "$SGLANG_COMMIT" > /opt/sglang_commit.txt

                # Install sglang from copied source with CUDA wheels
                RUN cd /sgl-workspace/sglang && \\
                    pip install --upgrade pip && \\
                    pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu121

                # Verify installation
                RUN python -c "import torch; print(f'PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"
            """))
        if not dockerfile_path: return commit_sha, False, "No Dockerfile"
        _apply_context_fixes(worktree_dir, dockerfile_path, project, commit_sha)
        tag = f"docker.io/{dockerhub_username}/{image_name}:{commit_sha}"
        cmd = [
            "docker", "buildx", "build",
            "--platform", platform,
            "--file", str(dockerfile_path),
            "--tag", tag,
            "--label", f"org.opencontainers.image.revision={commit_sha}",
            "--build-arg", f"SGLANG_COMMIT={commit_sha}",  # Pass commit for in-container proof
            "--no-cache"
        ]
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
    parser.add_argument("--dataset", default="nvidia-sglang-docker.jsonl")
    parser.add_argument("--blacklist", default="blacklist.txt")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--workdir", default="sglang")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-pushed", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--cache-dir", default=".buildx-cache")
    parser.add_argument("--show-build-logs", action="store_true")
    parser.add_argument("--project", default="sglang", choices=["vllm", "sglang"])
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
