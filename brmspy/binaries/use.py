import json
import tarfile
import tempfile
import urllib.request
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import rpy2.robjects as ro

from brmspy.binaries.config import get_active_runtime, set_active_runtime
from brmspy.binaries.env import system_fingerprint
from brmspy.binaries.github import get_github_asset_sha256_from_url
from brmspy.binaries.r import _get_r_pkg_installed, _try_force_unload_package, _r_namespace_loaded, _r_package_attached
from brmspy.helpers.log import greet, log, log_warning

OFFICIAL_RELEASE_PATTERN = "https://github.com/kaitumisuuringute-keskus/brmspy/"
HASH_FILENAME = "hash"
MANIFEST_FILENAME = "manifest.json"
RUNTIME_SUBDIR_NAME = "runtime"
TMP_EXTRACT_DIR_NAME = "_tmp_extract"

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
    
    This function reconfigures the running embedded R session to use an isolated 
    library environment from downloaded binaries. It does not mix or break the 
    default library tree already installed in the system.
    
    **What this function does:**
    
    1. Validates runtime bundle structure (manifest, Rlib, cmdstan directories)
    2. Optionally verifies system fingerprint matches bundle fingerprint
    3. Unloads brms, cmdstanr and rstan (if they are loaded)
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
        pass

    rlib_posix = rlib_dir.as_posix()
    cmdstan_posix = cmdstan_dir.as_posix()



    for pkg in ("brms", "cmdstanr", "rstan"):
        if _r_namespace_loaded(pkg) or _r_package_attached(pkg):
            _try_force_unload_package(pkg, uninstall=False)

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

    # Save the activated runtime path for auto-activation on next import
    try:
        from brmspy.binaries.config import set_active_runtime
        set_active_runtime(runtime_root)
    except Exception as e:
        log_warning(f"Failed to save runtime configuration: {e}")

    # At this point, R is configured to use the prebuilt runtime.
    # Any further brms/cmdstanr calls in this process will use it.

    greet()




def _ensure_base_dir(base_dir: Optional[Union[str, Path]]) -> Path:
    if base_dir is None:
        base_dir = Path.home() / ".brmspy" / "runtime"
    base_dir = Path(base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _runtime_root_for_system(base_dir: Path, runtime_version: str) -> Path:
    """
    Canonical installation path for a runtime:
    {base_dir}/{system_fingerprint}-{runtime_version}
    """
    sys_fp = system_fingerprint()
    return base_dir / f"{sys_fp}-{runtime_version}"


def _resolve_source(
    url: Optional[str], bundle: Optional[Union[str, Path]]
) -> Tuple[Path, Optional[Path]]:
    """
    Resolve the source into a local path.

    Returns
    -------
    (bundle_path, tmp_download)
      bundle_path : Path to archive or directory
      tmp_download : Path to temp file that should be cleaned up (or None)
    """
    if (url is None) == (bundle is None):
        raise ValueError("Exactly one of `url` or `bundle` must be provided.")

    tmp_download: Optional[Path] = None

    if url is not None:
        # Download into a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        log(f"Fetching {url} to {tmp_path}")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_download = tmp_path
        bundle_path = tmp_path
    else:
        bundle_path = Path(bundle).expanduser().resolve()  # type: ignore[arg-type]

    return bundle_path, tmp_download


def _is_runtime_dir(path: Path) -> bool:
    """
    Minimal structural check for a runtime directory.
    """
    if not path.is_dir():
        return False
    manifest = path / MANIFEST_FILENAME
    return manifest.is_file()


def _load_manifest(runtime_root: Path) -> dict:
    manifest_path = runtime_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing {MANIFEST_FILENAME} in {runtime_root}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_stored_hash(runtime_root: Path) -> Optional[str]:
    hash_path = runtime_root / HASH_FILENAME
    if not hash_path.is_file():
        return None
    return hash_path.read_text(encoding="utf-8").strip()


def _write_stored_hash(runtime_root: Path, hash_value: str) -> None:
    hash_path = runtime_root / HASH_FILENAME
    hash_path.write_text(hash_value.strip() + "\n", encoding="utf-8")


def _maybe_reuse_existing_runtime(
    runtime_root: Path,
    expected_hash: Optional[str],
    activate: bool,
) -> Optional[Path]:
    """
    If a runtime already exists and its stored hash matches `expected_hash`,
    reuse it and optionally activate it.
    """
    if not runtime_root.exists():
        return None
    if not _is_runtime_dir(runtime_root):
        return None
    if expected_hash is None:
        # No attested hash to compare against; let caller decide what to do.
        return None

    stored_hash = _read_stored_hash(runtime_root)
    if stored_hash is None:
        return None

    if stored_hash != expected_hash:
        log(
            f"Existing runtime at {runtime_root} has hash "
            f"{stored_hash}, expected {expected_hash}. Reinstalling."
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        return None

    log(f"Reusing existing runtime at {runtime_root}")
    if activate:
        activate_runtime(runtime_root)
    return runtime_root


def _install_from_archive(
    archive_path: Path,
    runtime_root: Path,
    runtime_version: str,
    base_dir: Path,
) -> Path:
    """
    Extract an archive produced by the build script and install it to runtime_root.
    Archive is expected to contain top-level 'runtime/' directory.
    """
    log(f"Extracting archive {archive_path}")

    temp_extract_root = base_dir / TMP_EXTRACT_DIR_NAME
    if temp_extract_root.exists():
        shutil.rmtree(temp_extract_root)
    temp_extract_root.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, mode="r:*") as tf:
            tf.extractall(path=temp_extract_root)

        runtime_tmp = temp_extract_root / RUNTIME_SUBDIR_NAME
        if not runtime_tmp.is_dir():
            raise RuntimeError(
                f"Extracted archive does not contain '{RUNTIME_SUBDIR_NAME}/' "
                f"under {temp_extract_root}"
            )

        manifest = _load_manifest(runtime_tmp)
        mf_runtime_version = manifest.get("runtime_version")
        if mf_runtime_version and mf_runtime_version != runtime_version:
            log_warning(
                f"manifest runtime_version={mf_runtime_version} "
                f"!= expected={runtime_version}"
            )

        if runtime_root.exists():
            log_warning(f"Removing existing runtime at {runtime_root}")
            shutil.rmtree(runtime_root, ignore_errors=True)

        log(f"Moving runtime to {runtime_root}")
        shutil.move(str(runtime_tmp), str(runtime_root))

    finally:
        shutil.rmtree(temp_extract_root, ignore_errors=True)

    return runtime_root


def _install_from_directory(
    src_dir: Path,
    runtime_root: Path,
    runtime_version: str,
) -> Path:
    """
    Install a runtime from an already-extracted directory.

    If src_dir is already the canonical runtime_root, just validate it.
    Otherwise move/copy into runtime_root.
    """
    manifest = _load_manifest(src_dir)
    mf_runtime_version = manifest.get("runtime_version")
    if mf_runtime_version and mf_runtime_version != runtime_version:
        log_warning(
            f"manifest runtime_version={mf_runtime_version} "
            f"!= expected={runtime_version}"
        )

    src_dir = src_dir.resolve()
    runtime_root = runtime_root.resolve()

    if src_dir == runtime_root:
        # Already in place
        return runtime_root

    if runtime_root.exists():
        log_warning(f"Removing existing runtime at {runtime_root}")
        shutil.rmtree(runtime_root, ignore_errors=True)

    log(f"Moving runtime from {src_dir} to {runtime_root}")
    shutil.move(str(src_dir), str(runtime_root))

    return runtime_root



def install_and_activate_runtime(
    url: Optional[str] = None,
    bundle: Optional[Union[str, Path]] = None,
    runtime_version: str = "0.1.0",
    base_dir: Optional[Union[str, Path]] = None,
    activate: bool = True,
    expected_hash: Optional[str] = None,
    require_attestation: bool = True,
) -> Path:
    """
    Download, install, and optionally activate a prebuilt brms runtime bundle.

    This is the high-level orchestration function. It:

      1. Resolves the source (URL → temp file, or local archive/dir)
      2. Determines the canonical runtime_root for this system+version
      3. Optionally reuses an existing runtime if the stored hash matches
      4. Installs from archive or directory into runtime_root
      5. Stores the expected hash (if provided) for fast future reuse
      6. Optionally activates the runtime in the current R session

    Parameters
    ----------
    url : str, optional
        URL to download runtime bundle archive (.tar.gz, .tar.bz2, etc.).
        Mutually exclusive with `bundle`.
    bundle : str or Path, optional
        Local path to runtime bundle, either:
        - Archive file (.tar.gz, .tar.bz2, etc.) to extract, or
        - Directory containing an extracted runtime (with manifest.json).
        Mutually exclusive with `url`.
    runtime_version : str, default="0.1.0"
        Logical runtime version used in the canonical runtime path
        {base_dir}/{system_fingerprint}-{runtime_version}.
    base_dir : str or Path, optional
        Base directory for runtime installation.
        Default: ~/.brmspy/runtime/
    activate : bool, default=True
        If True, call `activate_runtime()` after installation/reuse.
    expected_hash : str, optional
        Attested hash (e.g. sha256 from GitHub release assets). If provided,
        the installer will:
        - Reuse an existing runtime only if its stored hash matches this value.
        - Store the hash into {runtime_root}/hash after (re)install.
        If NOT provided, but the binary is an official kaitumisuuringute-keskus/brmspy 
        one, the expected_hash will be automatically fetched
    require_attestation : bool, default=True
        If True, `expected_hash` must be provided; otherwise a ValueError is
        raised. This lets callers enforce that only attested runtimes are used.

    Returns
    -------
    Path
        Path to the installed (or reused) runtime root directory.
    """

    if url and url.startswith(OFFICIAL_RELEASE_PATTERN) and not expected_hash:
        auto_hash = get_github_asset_sha256_from_url(url, require_digest=True)
        if expected_hash is not None and expected_hash != auto_hash:
            raise ValueError(
                "Expected hash does not match attested GitHub digest for "
                f"{url!r}: expected {expected_hash}, got {auto_hash}"
            )
        expected_hash = auto_hash

    if require_attestation and expected_hash is None:
        raise ValueError(
            "require_attestation=True but no expected_hash was provided."
        )

    base_dir_path = _ensure_base_dir(base_dir)
    runtime_root = _runtime_root_for_system(base_dir_path, runtime_version)

    # Fast path: reuse if hash matches
    reused = _maybe_reuse_existing_runtime(
        runtime_root=runtime_root,
        expected_hash=expected_hash,
        activate=activate,
    )
    if reused is not None:
        return reused

    # Resolve source and install
    bundle_path, tmp_download = _resolve_source(url, bundle)
    try:
        if bundle_path.is_dir():
            runtime_root = _install_from_directory(
                src_dir=bundle_path,
                runtime_root=runtime_root,
                runtime_version=runtime_version,
            )
        else:
            runtime_root = _install_from_archive(
                archive_path=bundle_path,
                runtime_root=runtime_root,
                runtime_version=runtime_version,
                base_dir=base_dir_path,
            )
    finally:
        # Clean up downloaded temp file if any
        if tmp_download is not None and tmp_download.exists():
            tmp_download.unlink()

    # Store attested hash for future quick startups
    if expected_hash is not None:
        _write_stored_hash(runtime_root, expected_hash)

    # Final sanity check (manifest present etc.)
    _ = _load_manifest(runtime_root)

    if activate:
        log(f"Activating runtime at {runtime_root}")
        activate_runtime(runtime_root)

    return runtime_root


def autoload_last_runtime():
    # Auto-activate saved prebuilt runtime if available
    try:
        runtime_path = get_active_runtime()
        if runtime_path is not None and runtime_path.exists():
            try:
                activate_runtime(runtime_path)
            except Exception as e:
                log_warning(f"Failed to auto-activate saved runtime at {runtime_path}: {e}")
                set_active_runtime(None)
    except Exception:
        # Silently skip if config module unavailable or other issues
        pass