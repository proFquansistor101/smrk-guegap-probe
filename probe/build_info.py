# build info module
# SMRK-GUEGAP Probe — build_info.py
# Minimal, audit-friendly build/platform metadata capture (v1)

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_git_commit(repo_root: str) -> Optional[str]:
    """
    Returns the current git commit hash if available, otherwise None.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        return None
    return None


def _try_git_is_dirty(repo_root: str) -> Optional[bool]:
    """
    Returns True/False if git is available; otherwise None.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return len(out.strip()) > 0
    except Exception:
        return None


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def get_platform_info() -> Dict[str, Any]:
    """
    Platform metadata for output artifacts.
    Keep this lightweight and non-invasive.
    """
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "endianness": sys.byteorder,
        # float mode is assumed IEEE754 on mainstream platforms; record as best-effort
        "float_mode": "IEEE754" if sys.float_info.radix == 2 else "unknown",
    }


def get_dependencies_info() -> Dict[str, Any]:
    """
    Dependency versions (best-effort).
    """
    return {
        "python": platform.python_version(),
        "numpy": _safe_import_version("numpy"),
        "scipy": _safe_import_version("scipy"),
        "jsonschema": _safe_import_version("jsonschema"),
    }


def get_reference_hashes(reference_dir: str) -> Dict[str, Any]:
    """
    Computes SHA256 hashes for frozen reference distribution files.
    If files are missing, returns null entries.
    """
    files = {
        "gue_r_reference_sha256": os.path.join(reference_dir, "r_ref_gue_v1.json"),
        "goe_r_reference_sha256": os.path.join(reference_dir, "r_ref_goe_v1.json"),
        "poisson_r_reference_sha256": os.path.join(reference_dir, "r_ref_poisson_v1.json"),
    }

    out: Dict[str, Any] = {}
    for key, path in files.items():
        out[key] = _sha256_file(path) if os.path.isfile(path) else None
    return out


def get_build_info(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Build metadata for input/output artifacts.

    repo_root:
      Path to repo root (recommended). If None, attempts to infer:
      this file is in <repo>/probe/build_info.py => repo_root = parent of probe/.
    """
    if repo_root is None:
        # infer repo root: .../probe/build_info.py -> .../
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    reference_dir = os.path.join(repo_root, "reference")

    return {
        "git_commit": _try_git_commit(repo_root),
        "git_dirty": _try_git_is_dirty(repo_root),
        "repo_root": os.path.basename(repo_root),
        "dependencies": get_dependencies_info(),
        "reference_data": get_reference_hashes(reference_dir),
    }


if __name__ == "__main__":
    # Quick local debug: print as JSON
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = {
        "build": get_build_info(repo_root),
        "platform": get_platform_info(),
    }
    print(json.dumps(data, indent=2, sort_keys=True))
