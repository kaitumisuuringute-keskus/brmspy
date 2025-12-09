"""
Stage runtime tree structure.
"""

import json
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

from brmspy._runtime._platform import system_fingerprint
from brmspy.helpers.log import log


def _generate_manifest_hash(manifest: dict) -> str:
    """Generate deterministic SHA256 hash of runtime manifest."""
    manifest_string = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(manifest_string.encode('utf-8')).hexdigest()


def stage_runtime_tree(base_dir: Path, metadata: dict, runtime_version: str) -> Path:
    """
    Create runtime directory structure and copy all required files.
    
    Builds the complete runtime directory tree by:
    1. Creating fingerprint-specific directory structure
    2. Copying all R packages to Rlib/
    3. Copying CmdStan installation to cmdstan/
    4. Generating manifest.json with checksums
    
    Parameters
    ----------
    base_dir : Path
        Base directory for runtime tree
    metadata : dict
        Metadata from collect_runtime_metadata()
    runtime_version : str
        Runtime schema version (e.g., "0.1.0")
    
    Returns
    -------
    Path
        Path to the runtime root directory
    """
    fingerprint = system_fingerprint()
    if fingerprint is None:
        raise RuntimeError("system_fingerprint() returned None; cannot build runtime bundle.")

    runtime_root = base_dir / f"{fingerprint}-{runtime_version}"
    rlib_dir = runtime_root / "Rlib"
    cmdstan_dir = runtime_root / "cmdstan"

    runtime_root.mkdir(parents=True, exist_ok=True)
    rlib_dir.mkdir(parents=True, exist_ok=True)

    # Copy R packages into Rlib/
    pkgs = metadata.get("packages", [])
    if not pkgs:
        raise RuntimeError("No package metadata returned from R; cannot build runtime.")

    for pkg in pkgs:
        name = pkg["Package"]
        libpath = pkg["LibPath"]
        src = Path(libpath) / name
        dest = rlib_dir / name

        if not src.exists():
            raise RuntimeError(f"Package directory not found: {src}")

        log(f"[Rlib] Copying {name} from {src} to {dest}")
        shutil.copytree(src, dest, dirs_exist_ok=True)

    # Copy CmdStan tree into cmdstan/
    cmdstan_path = Path(metadata["cmdstan_path"])
    if not cmdstan_path.exists():
        raise RuntimeError(f"cmdstan_path does not exist on disk: {cmdstan_path}")

    log(f"[cmdstan] Copying CmdStan from {cmdstan_path} to {cmdstan_dir}")
    shutil.copytree(cmdstan_path, cmdstan_dir, dirs_exist_ok=True)


    # On macOS, drop precompiled headers (PCH) to avoid SDK/PCH mismatch issues
    if sys.platform == "darwin":
        model_dir = cmdstan_dir / "stan" / "src" / "stan" / "model"
        if model_dir.exists():
            for entry in model_dir.iterdir():
                # Clang stores PCH in dirs like "model_header.hpp.gch/"
                if entry.is_dir() and entry.name.endswith(".hpp.gch"):
                    log(f"[cmdstan] Removing PCH directory on macOS: {entry}")
                    shutil.rmtree(entry, ignore_errors=True)

    # Write manifest.json
    r_pkg_versions = {pkg["Package"]: pkg["Version"] for pkg in pkgs}

    manifest = {
        "runtime_version": runtime_version,
        "fingerprint": fingerprint,
        "r_version": metadata["r_version"],
        "cmdstan_version": metadata["cmdstan_version"],
        "r_packages": r_pkg_versions
    }

    hash_val = _generate_manifest_hash(manifest)
    manifest['manifest_hash'] = hash_val
    manifest['built_at'] = datetime.now(timezone.utc).isoformat()

    manifest_path = runtime_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"[manifest] Wrote {manifest_path}")

    return runtime_root