import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import List, cast

import rpy2.robjects as ro

from brmspy.binaries.env import system_fingerprint

def _generate_manifest_hash(manifest):
    # Dump to JSON string with sorted keys to ensure determinism
    # separators=(',', ':') removes whitespace to keep it compact and consistent
    manifest_string = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    
    return hashlib.sha256(manifest_string.encode('utf-8')).hexdigest()

def _run_r_json(code: str) -> dict:
    """
    Run R code that returns a single JSON string and parse it to Python dict.
    The last expression in `code` MUST evaluate to a character vector of length 1.
    """
    res = cast(List[str], ro.r(code))
    # rpy2 will return a StrVector; index 0 to get the Python string.
    json_str = res[0]
    return json.loads(json_str)

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

def collect_runtime_metadata() -> dict:
    """
    Ask R (via rpy2) for:
      - R version
      - cmdstan path + version
      - full dependency closure of brms + cmdstanr
      - per-package info (Package, Version, LibPath, Priority)
    Returns a Python dict.
    """
    with open(os.path.join(script_dir, "build-manifest.R"), "r") as f:
        r_code = f.read()
    # Make sure jsonlite is available
    ro.r('if (!requireNamespace("jsonlite", quietly = TRUE)) '
         'install.packages("jsonlite", repos="https://cloud.r-project.org")')

    return _run_r_json(r_code)


def stage_runtime_tree(base_dir: Path, metadata: dict, runtime_version: str) -> Path:
    """
    Create the runtime directory tree:
      base_dir / fingerprint / { manifest.json, Rlib/, cmdstan/ }

    Returns the path to the runtime root directory.
    """
    fingerprint = system_fingerprint()
    if fingerprint is None:
        raise RuntimeError("system_fingerprint() returned None; cannot build runtime bundle.")

    runtime_root = base_dir / fingerprint
    rlib_dir = runtime_root / "Rlib"
    cmdstan_dir = runtime_root / "cmdstan"

    runtime_root.mkdir(parents=True, exist_ok=True)
    rlib_dir.mkdir(parents=True, exist_ok=True)

    # ---- Copy R packages into Rlib/ ----
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

        print(f"[Rlib] Copying {name} from {src} → {dest}")
        # Python 3.8+: dirs_exist_ok=True
        shutil.copytree(src, dest, dirs_exist_ok=True)

    # ---- Copy CmdStan tree into cmdstan/ ----
    cmdstan_path = Path(metadata["cmdstan_path"])
    if not cmdstan_path.exists():
        raise RuntimeError(f"cmdstan_path does not exist on disk: {cmdstan_path}")

    print(f"[cmdstan] Copying CmdStan from {cmdstan_path} → {cmdstan_dir}")
    shutil.copytree(cmdstan_path, cmdstan_dir, dirs_exist_ok=True)

    # ---- Write manifest.json ----
    # Build packages mapping: name -> version
    r_pkg_versions = {pkg["Package"]: pkg["Version"] for pkg in pkgs}

    manifest = {
        "runtime_version": runtime_version,
        "fingerprint": fingerprint,
        "r_version": metadata["r_version"],
        "cmdstan_version": metadata["cmdstan_version"],
        "r_packages": r_pkg_versions
    }

    hash = _generate_manifest_hash(manifest)
    manifest['manifest_hash'] = hash
    manifest['built_at'] = datetime.now(timezone.utc).isoformat()

    manifest_path = runtime_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] Wrote {manifest_path}")

    return runtime_root


def pack_runtime(runtime_root: Path, out_dir: Path, runtime_version: str) -> Path:
    """
    Create a .tar.gz archive from runtime_root.
    Archive name: brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz
    """
    fingerprint = runtime_root.name  # since we used base_dir / fingerprint
    archive_name = f"brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz"
    archive_path = out_dir / archive_name

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tar] Creating archive {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tf:
        # Add the runtime root directory contents under "runtime/"
        tf.add(runtime_root, arcname="runtime")

    return archive_path


def main():
    parser = argparse.ArgumentParser(description="Build brmspy prebuilt runtime bundle.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runtime_build",
        help="Directory where the runtime tree and archive will be written.",
    )
    parser.add_argument(
        "--runtime-version",
        type=str,
        default="0.1.0",
        help="Logical runtime schema/version identifier (not necessarily pip version).",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir).resolve()
    out_dir = base_dir  # can separate if you want
    runtime_version = args.runtime_version

    # Set the CRAN mirror globally for this session. 
    # This prevents 'build-manifest.R' or subsequent installs from prompting for a mirror.
    ro.r('options(repos = c(CRAN = "https://cloud.r-project.org"))')

    print("[meta] Collecting R / brms / cmdstanr metadata via rpy2...")
    metadata = collect_runtime_metadata()

    print("[stage] Staging runtime tree...")
    runtime_root = stage_runtime_tree(base_dir, metadata, runtime_version)

    print("[pack] Packing runtime to tar.gz...")
    archive_path = pack_runtime(runtime_root, out_dir, runtime_version)

    print(f"[done] Runtime bundle created: {archive_path}")


if __name__ == "__main__":
    main()
