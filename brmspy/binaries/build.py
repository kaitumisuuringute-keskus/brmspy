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
from brmspy.helpers.log import log


def _generate_manifest_hash(manifest):
    """
    Generate deterministic SHA256 hash of runtime manifest.
    
    Creates a cryptographic hash from the manifest dictionary to
    verify runtime bundle integrity and detect corruption. The hash
    is deterministic across platforms by using sorted JSON serialization.
    
    Parameters
    ----------
    manifest : dict
        Runtime manifest dictionary containing:
        - runtime_version : str - Runtime schema version
        - fingerprint : str - Platform fingerprint
        - r_version : str - R version
        - cmdstan_version : str - CmdStan version
        - r_packages : dict - Mapping of package names to versions
    
    Returns
    -------
    str
        64-character hexadecimal SHA256 hash
    
    Examples
    --------

    ```
    manifest = {
        "runtime_version": "0.1.0",
        "fingerprint": "linux-x86_64-r4.3",
        "r_version": "4.3.1",
        "cmdstan_version": "2.33.1",
        "r_packages": {"brms": "2.21.0", "cmdstanr": "0.8.1"}
    }
    hash_val = _generate_manifest_hash(manifest)
    len(hash_val)  # 64
    ```

    """
    # Dump to JSON string with sorted keys to ensure determinism
    # separators=(',', ':') removes whitespace to keep it compact and consistent
    manifest_string = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    
    return hashlib.sha256(manifest_string.encode('utf-8')).hexdigest()

def _run_r_json(code: str) -> dict:
    """
    Execute R code and parse JSON output to Python dictionary.
    
    Runs R code via rpy2 that must return a single JSON string, then
    parses it to a Python dict. Used for extracting structured data
    from R in a type-safe way.
    
    Parameters
    ----------
    code : str
        R code to execute. The last expression MUST evaluate to
        a character vector of length 1 containing valid JSON.
    
    Returns
    -------
    dict
        Parsed JSON as Python dictionary
    
    Raises
    ------
    json.JSONDecodeError
        If R output is not valid JSON
    IndexError
        If R returns empty vector
    
    Examples
    --------

    ```python
    r_code = '''
    library(jsonlite)
    toJSON(list(
        r_version = as.character(getRversion()),
        packages = installed.packages()[1:2, "Package"]
    ))
    '''
    result = _run_r_json(r_code)
    print(result["r_version"])
    ```

    """
    res = cast(List[str], ro.r(code))
    # rpy2 will return a StrVector; index 0 to get the Python string.
    json_str = res[0]
    return json.loads(json_str)

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

def collect_runtime_metadata() -> dict:
    """
    Collect comprehensive R environment metadata for runtime bundle.
    
    Queries R via rpy2 to gather complete information about the current
    R installation, including R version, CmdStan installation, and full
    dependency closure of brms + cmdstanr with all package details.
    
    Returns
    -------
    dict
        Metadata dictionary containing:
        - r_version : str - R version (e.g., "4.3.1")
        - cmdstan_path : str - Path to CmdStan installation
        - cmdstan_version : str - CmdStan version
        - packages : list of dict - Package information with fields:
            - Package : str - Package name
            - Version : str - Package version
            - LibPath : str - Installation library path
            - Priority : str - Package priority level
    
    Raises
    ------
    RuntimeError
        If required R packages (jsonlite) cannot be installed
        If build-manifest.R script cannot be read
    
    Notes
    -----
    This function executes the build-manifest.R script which uses
    R's package dependency resolution to find the complete dependency
    closure. It ensures all transitive dependencies are included.
    
    Examples
    --------

    ```python
    metadata = collect_runtime_metadata()
    print(f"R version: {metadata['r_version']}")
    print(f"CmdStan: {metadata['cmdstan_version']}")
    print(f"Packages: {len(metadata['packages'])}")
    
    # Check if brms is included
    pkg_names = [p["Package"] for p in metadata["packages"]]
    assert "brms" in pkg_names
    ```

    """
    with open(os.path.join(script_dir, "build-manifest.R"), "r") as f:
        r_code = f.read()
    # Make sure jsonlite is available
    ro.r('if (!requireNamespace("jsonlite", quietly = TRUE)) '
         'install.packages("jsonlite", repos="https://cloud.r-project.org")')

    return _run_r_json(r_code)


def stage_runtime_tree(base_dir: Path, metadata: dict, runtime_version: str) -> Path:
    """
    Create runtime directory structure and copy all required files.
    
    Builds the complete runtime directory tree by:
    1. Creating fingerprint-specific directory structure
    2. Copying all R packages to Rlib/
    3. Copying CmdStan installation to cmdstan/
    4. Generating manifest.json with checksums
    
    The resulting structure:
    base_dir/
      {fingerprint}/
        manifest.json
        Rlib/
          {package1}/
          {package2}/
          ...
        cmdstan/
          {cmdstan files}
    
    Parameters
    ----------
    base_dir : Path
        Base directory for runtime tree
    metadata : dict
        Metadata from collect_runtime_metadata() containing:
        - packages : list - R package information
        - cmdstan_path : str - CmdStan location
        - r_version : str
        - cmdstan_version : str
    runtime_version : str
        Runtime schema version (e.g., "0.1.0")
    
    Returns
    -------
    Path
        Path to the runtime root directory (base_dir/fingerprint)
    
    Raises
    ------
    RuntimeError
        If system_fingerprint() returns None
        If no package metadata provided
        If package directories don't exist
        If cmdstan_path doesn't exist
    
    Notes
    -----
    The fingerprint is determined by system_fingerprint() which includes:
    - OS (linux/macos/windows)
    - Architecture (x86_64/arm64)
    - R version (major.minor)
    
    The manifest.json includes a SHA256 hash for integrity verification.
    
    Examples
    --------

    ```python
    from pathlib import Path
    
    metadata = collect_runtime_metadata()
    base = Path("./runtime_build")
    
    runtime_root = stage_runtime_tree(
        base,
        metadata,
        runtime_version="0.1.0"
    )
    
    print(f"Runtime staged at: {runtime_root}")
    print(f"Manifest: {runtime_root / 'manifest.json'}")
    print(f"R packages: {runtime_root / 'Rlib'}")
    print(f"CmdStan: {runtime_root / 'cmdstan'}")
    ```

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

        log(f"[Rlib] Copying {name} from {src} to {dest}")
        # Python 3.8+: dirs_exist_ok=True
        shutil.copytree(src, dest, dirs_exist_ok=True)

    # ---- Copy CmdStan tree into cmdstan/ ----
    cmdstan_path = Path(metadata["cmdstan_path"])
    if not cmdstan_path.exists():
        raise RuntimeError(f"cmdstan_path does not exist on disk: {cmdstan_path}")

    log(f"[cmdstan] Copying CmdStan from {cmdstan_path} to {cmdstan_dir}")
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
    log(f"[manifest] Wrote {manifest_path}")

    return runtime_root


def pack_runtime(runtime_root: Path, out_dir: Path, runtime_version: str) -> Path:
    """
    Create compressed tar archive from runtime directory.
    
    Packages the staged runtime directory into a distributable .tar.gz
    archive with standardized naming for platform/version identification.
    
    Archive naming format:
    brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz
    
    For example:
    brmspy-runtime-0.1.0-linux-x86_64-r4.3.tar.gz
    
    Parameters
    ----------
    runtime_root : Path
        Path to staged runtime directory (from stage_runtime_tree)
    out_dir : Path
        Output directory for archive file
    runtime_version : str
        Runtime schema version (e.g., "0.1.0")
    
    Returns
    -------
    Path
        Path to created .tar.gz archive file
    
    Notes
    -----
    The archive contains a top-level "runtime/" directory with all
    runtime files. This structure is expected by install_and_activate_runtime().
    
    The fingerprint is extracted from runtime_root.name, which should
    match the fingerprint used in stage_runtime_tree().
    
    Examples
    --------

    ```python
    from pathlib import Path
    
    runtime_root = Path("./runtime_build/linux-x86_64-r4.3")
    out_dir = Path("./dist")
    
    archive = pack_runtime(
        runtime_root,
        out_dir,
        runtime_version="0.1.0"
    )
    
    print(f"Archive created: {archive}")
    print(f"Size: {archive.stat().st_size / 1024 / 1024:.1f} MB")
    ```

    """
    fingerprint = runtime_root.name  # since we used base_dir / fingerprint
    archive_name = f"brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz"
    archive_path = out_dir / archive_name

    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[tar] Creating archive {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tf:
        # Add the runtime root directory contents under "runtime/"
        tf.add(runtime_root, arcname="runtime")

    return archive_path


def main():
    """
    CLI entry point for building brmspy prebuilt runtime bundles.
    
    Command-line tool that orchestrates the complete runtime build process:
    1. Collect R environment metadata
    2. Stage runtime directory tree
    3. Pack into distributable archive
    
    Command-line Arguments
    ----------------------
    --output-dir : str, default="runtime_build"
        Directory where runtime tree and archive will be written
    --runtime-version : str, default="0.1.0"
        Runtime schema version identifier (not pip version)
    
    Examples
    --------
    Build with defaults:

    ```bash
    python -m brmspy.binaries.build
    ```

    Specify custom output directory and version:
    
    ```bash
    python -m brmspy.binaries.build \
        --output-dir /tmp/runtime \
        --runtime-version 0.2.0
    ```

    Notes
    -----
    Requires:
    - R with brms and cmdstanr installed
    - CmdStan compiled and configured
    - Write permissions to output directory
    
    The build process can take 5-10 minutes depending on the number
    of dependencies and disk speed.
    
    See Also
    --------
    collect_runtime_metadata : Gathers R environment info
    stage_runtime_tree : Creates runtime directory structure
    pack_runtime : Creates distributable archive
    """
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

    log("[meta] Collecting R / brms / cmdstanr metadata via rpy2...")
    metadata = collect_runtime_metadata()

    log("[stage] Staging runtime tree...")
    runtime_root = stage_runtime_tree(base_dir, metadata, runtime_version)

    log("[pack] Packing runtime to tar.gz...")
    archive_path = pack_runtime(runtime_root, out_dir, runtime_version)

    log(f"[done] Runtime bundle created: {archive_path}")


if __name__ == "__main__":
    main()
