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
    Activate a prebuilt brms runtime bundle in the current R session.
    
    Configures the embedded R session (via rpy2) to use libraries and
    CmdStan from a prebuilt runtime bundle. Updates R's library paths
    and sets cmdstanr's CmdStan path, then performs sanity checks.
    
    Parameters
    ----------
    runtime_root : str or Path
        Path to runtime bundle root directory containing:
        - manifest.json: Bundle metadata and fingerprint
        - Rlib/: R package library with brms, cmdstanr, etc.
        - cmdstan/: CmdStan installation
    
    Raises
    ------
    RuntimeError
        If required directories/files are missing or validation fails
    
    Notes
    -----
    **Important Limitations:**
    
    This function reconfigures the running embedded R session but does NOT
    fully restart R (rpy2 cannot do that). For a truly clean environment:
    - Call this function BEFORE loading any R packages
    - Or restart the Python process after installation
    
    **What this function does:**
    
    1. Validates runtime bundle structure (manifest, Rlib, cmdstan directories)
    2. Optionally verifies system fingerprint matches bundle fingerprint
    3. Replaces R's .libPaths() with runtime Rlib directory
    4. Sets cmdstanr::set_cmdstan_path() to runtime cmdstan directory
    5. Performs sanity checks (brms and cmdstanr load successfully)
    
    **Fingerprint Validation:**
    
    If available, validates that the runtime bundle's fingerprint (from
    manifest.json) matches the current system fingerprint. This prevents
    incompatible bundles (e.g., Linux bundle on macOS) from being activated.
    
    Examples
    --------

    ```python
    from pathlib import Path
    from brmspy.binaries.use import activate_runtime
    
    # Activate a previously installed runtime
    runtime_path = Path.home() / ".brmspy" / "runtime" / "linux-x86_64-r4.3"
    activate_runtime(runtime_path)
    
    # Now brms and cmdstanr use the prebuilt bundle
    from brmspy import fit
    
    result = fit("y ~ x", data={"y": [1, 2, 3], "x": [1, 2, 3]})
    ```

    ```python
    # Activate runtime before any R operations (recommended)
    import sys
    from brmspy.binaries.use import activate_runtime
    
    # Do this FIRST, before importing brmspy or using rpy2
    activate_runtime("/path/to/runtime")
    
    # Now safe to use brmspy
    from brmspy import fit
    ```

    See Also
    --------
    install_and_activate_runtime : Download, install, and activate runtime
    brmspy.binaries.env.system_fingerprint : Get current system identifier
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
    ro.r(f'.libPaths(c("{rlib_posix}"))')

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
    Download, install, and optionally activate a prebuilt brms runtime bundle.
    
    Flexible installation function that can download from URL, extract from
    local archive, or use an already-extracted runtime directory. Installs
    to a fingerprint-specific directory and optionally activates immediately.
    
    Parameters
    ----------
    url : str, optional
        URL to download runtime bundle archive (.tar.gz, .tar.bz2, etc.)
        Mutually exclusive with `bundle`
    bundle : str or Path, optional
        Local path to runtime bundle, either:
        - Archive file (.tar.gz, .tar.bz2, etc.) to extract
        - Directory containing extracted runtime
        Mutually exclusive with `url`
    runtime_version : str, default="0.1.0"
        Expected runtime version for validation
    base_dir : str or Path, optional
        Base directory for runtime installation
        Default: ~/.brmspy/runtime/
        Runtime will be installed to: {base_dir}/{fingerprint}/
    activate : bool, default=True
        If True, call `activate_runtime()` after installation
    
    Returns
    -------
    Path
        Path to installed runtime root directory
    
    Raises
    ------
    ValueError
        If both or neither of `url` and `bundle` are provided
    RuntimeError
        If bundle structure is invalid or extraction fails
    
    Notes
    -----
    **Installation Process:**
    
    1. **Source Resolution:**
       - If `url`: Downloads to temporary file
       - If `bundle` (archive): Uses local archive
       - If `bundle` (directory): Assumes already extracted
    
    2. **Extraction (for archives):**
       - Extracts to temporary directory under base_dir
       - Expects archive structure: runtime/{manifest.json,Rlib/,cmdstan/}
       - Reads fingerprint from manifest.json
    
    3. **Installation:**
       - Moves runtime to: {base_dir}/{fingerprint}/
       - If fingerprint directory exists, removes it first
       - Cleans up temporary files
    
    4. **Validation:**
       - Verifies manifest.json exists and contains fingerprint
       - Optionally warns if runtime_version mismatch
    
    5. **Activation (if requested):**
       - Calls `activate_runtime()` to configure R session
    
    **Directory Structure:**
    
    After installation:
    ```
    ~/.brmspy/runtime/
    ├── linux-x86_64-r4.3/
    │   ├── manifest.json
    │   ├── Rlib/
    │   │   ├── brms/
    │   │   ├── cmdstanr/
    │   │   └── ... (other R packages)
    │   └── cmdstan/
    │       ├── bin/
    │       └── ... (CmdStan files)
    └── macos-arm64-r4.4/
        └── ... (another runtime)
    ```
    
    Examples
    --------

    ```python
    from brmspy.binaries.use import install_and_activate_runtime
    
    # Install from URL (e.g., GitHub Releases)
    url = "https://github.com/user/repo/releases/download/v1.0/runtime.tar.gz"
    runtime_path = install_and_activate_runtime(url=url)
    print(f"Installed to: {runtime_path}")
    
    # Now brms is ready to use
    from brmspy import fit
    result = fit("y ~ x", data={"y": [1, 2, 3], "x": [1, 2, 3]})
    ```

    ```python
    # Install from local archive without activating
    from pathlib import Path
    
    bundle_file = Path("/tmp/runtime-linux-x86_64-r4.3.tar.gz")
    runtime_path = install_and_activate_runtime(
        bundle=bundle_file,
        activate=False  # Don't activate yet
    )
    
    # Later, manually activate
    from brmspy.binaries.use import activate_runtime
    activate_runtime(runtime_path)
    ```

    ```python
    # Use already-extracted runtime directory
    extracted_dir = Path("/path/to/extracted/runtime")
    runtime_path = install_and_activate_runtime(
        bundle=extracted_dir,
        base_dir=Path.home() / "custom_runtimes"
    )
    ```

    ```python
    # Download and install with custom base directory
    runtime_path = install_and_activate_runtime(
        url="https://example.com/runtime.tar.gz",
        base_dir="/opt/brmspy/runtimes",
        runtime_version="1.0.0"
    )
    ```

    See Also
    --------
    activate_runtime : Activate an installed runtime
    brmspy.binaries.env.system_fingerprint : Get system fingerprint
    brmspy.binaries.build.pack_runtime : Create runtime bundles
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
