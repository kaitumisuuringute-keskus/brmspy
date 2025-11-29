import json
import tarfile
import tempfile
import urllib.request
import shutil
from pathlib import Path
from typing import Optional, Union

import rpy2.robjects as ro

def activate_runtime(runtime_root: Union[str, Path]) -> None:
    """
    Given a runtime root directory (containing manifest.json, Rlib/, cmdstan/),
    hook it into the current R session via rpy2:
      - prepend Rlib/ to .libPaths()
      - set cmdstanr::set_cmdstan_path()
      - do basic sanity checks

    This does NOT fully "restart R" (rpy2 cannot do that); it reconfigures
    the running embedded R. For a truly clean environment, call this before
    loading any R packages, or restart the Python process after installation.
    """
    runtime_root = Path(runtime_root).expanduser().resolve()

    manifest_path = runtime_root / "manifest.json"
    rlib_dir = runtime_root / "Rlib"
    cmdstan_dir = runtime_root / "cmdstan"

    if not manifest_path.is_file():
        raise RuntimeError(f"manifest.json not found in {runtime_root}")
    if not rlib_dir.is_dir():
        raise RuntimeError(f"Rlib directory not found in {runtime_root}")
    if not cmdstan_dir.is_dir():
        raise RuntimeError(f"cmdstan directory not found in {runtime_root}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Optional: verify fingerprint matches this system
    try:
        from .env import system_fingerprint  # adjust import to your layout
        expected_fp = system_fingerprint()
        mf_fp = manifest.get("fingerprint")
        if expected_fp and mf_fp and expected_fp != mf_fp:
            raise RuntimeError(
                f"Runtime fingerprint mismatch: "
                f"manifest={mf_fp}, system={expected_fp}"
            )
    except ImportError:
        # If envdetect isn't available here, you can skip this check
        pass

    rlib_posix = rlib_dir.as_posix()
    cmdstan_posix = cmdstan_dir.as_posix()

    # Alternative, more error prone:
        # Prepend Rlib to .libPaths()
        #ro.r(f'.libPaths(c("{rlib_posix}", .libPaths()))')

    # Replace libPaths
    ro.r(f'.libPaths(c("{rlib_posix}")')

    # Point cmdstanr to this cmdstan installation
    ro.r(
        f'''
        if (!requireNamespace("cmdstanr", quietly = TRUE)) {{
          stop("cmdstanr is not available in the runtime Rlib: {rlib_posix}")
        }}
        cmdstanr::set_cmdstan_path("{cmdstan_posix}")
        '''
    )

    # Basic sanity checks: can we load brms/cmdstanr and query versions?
    ro.r(
        '''
        if (!requireNamespace("brms", quietly = TRUE)) {
          stop("brms is not available in the runtime Rlib.")
        }
        invisible(cmdstanr::cmdstan_version())
        '''
    )

    # At this point, R is configured to use the prebuilt runtime.
    # Any further brms/cmdstanr calls in this process will use it.


def install_and_activate_runtime(
    url: Optional[str] = None,
    bundle: Optional[Union[str, Path]] = None,
    runtime_version: str = "0.1.0",
    base_dir: Optional[Union[str, Path]] = None,
    activate: bool = True,
) -> Path:
    """
    Install (and optionally activate) a prebuilt runtime.

    - If `url` is provided, downloads a tar archive (e.g. .tar.gz) and extracts it.
    - If `bundle` is provided:
        * if it's a tar.* file -> extracts it
        * if it's a directory -> assumes it's already an extracted runtime root
    - Extracts into:
        base_dir / fingerprint
      where `fingerprint` is taken from manifest.json.
    - Optionally calls `activate_runtime()`.

    Returns: Path to the runtime root directory.
    """
    if (url is None) == (bundle is None):
        raise ValueError("Exactly one of `url` or `bundle` must be provided.")

    if base_dir is None:
        base_dir = Path.home() / ".brmspy" / "runtime"
    else:
        base_dir = Path(base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    tmp_download: Optional[Path] = None

    # 1) Resolve source (URL → temp file, or local path)
    if url is not None:
        # Download into a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        print(f"[download] Fetching {url} → {tmp_path}")
        urllib.request.urlretrieve(url, tmp_path)
        bundle_path = tmp_path
        tmp_download = tmp_path
    elif bundle:
        bundle_path = Path(bundle).expanduser().resolve()
    else:
        raise Exception("url is none and bundle is none")

    # 2) If the bundle is already a directory with manifest.json, just use it
    if bundle_path.is_dir():
        manifest_path = bundle_path / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Directory {bundle_path} does not look like a runtime root "
                f"(missing manifest.json)."
            )
        runtime_root = bundle_path

    else:
        # 3) Assume a tar.* archive containing a top-level "runtime/" directory
        #    (as produced by the build script).
        print(f"[extract] Extracting archive {bundle_path}")
        with tarfile.open(bundle_path, mode="r:*") as tf:
            # Extract to a temp directory under base_dir first
            temp_extract_root = base_dir / "_tmp_extract"
            if temp_extract_root.exists():
                shutil.rmtree(temp_extract_root)
            temp_extract_root.mkdir(parents=True, exist_ok=True)

            tf.extractall(path=temp_extract_root)

        # Expect temp_extract_root / "runtime"
        runtime_tmp = temp_extract_root / "runtime"
        manifest_path = runtime_tmp / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Extracted archive does not contain runtime/manifest.json "
                f"under {temp_extract_root}"
            )

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        fingerprint = manifest.get("fingerprint")
        if not fingerprint:
            raise RuntimeError("manifest.json is missing 'fingerprint' field.")

        # Final runtime_root = base_dir / fingerprint
        runtime_root = base_dir / fingerprint

        if runtime_root.exists():
            print(f"[extract] Removing existing runtime at {runtime_root}")
            shutil.rmtree(runtime_root)

        print(f"[extract] Moving runtime → {runtime_root}")
        shutil.move(str(runtime_tmp), str(runtime_root))

        # Clean temporary extraction dir
        shutil.rmtree(temp_extract_root, ignore_errors=True)

    # Clean up downloaded temp file if any
    if tmp_download is not None and tmp_download.exists():
        tmp_download.unlink()

    # Optional: sanity check that manifest runtime_version matches expectation
    manifest_path = runtime_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    mf_runtime_version = manifest.get("runtime_version")
    if mf_runtime_version and mf_runtime_version != runtime_version:
        print(
            f"[warn] manifest runtime_version={mf_runtime_version} "
            f"!= expected={runtime_version}"
        )

    # 4) Hook it into R if requested
    if activate:
        print(f"[activate] Activating runtime at {runtime_root}")
        activate_runtime(runtime_root)

    return runtime_root
