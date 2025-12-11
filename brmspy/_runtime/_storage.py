"""
Runtime directory operations. Disk only, no R, no config.
"""

import shutil
import tarfile
from pathlib import Path

from brmspy._runtime._platform import get_os


def get_runtime_base_dir() -> Path:
    """Returns ~/.brmspy/runtime/, creating if needed."""
    base_dir = Path.home() / ".brmspy" / "runtime"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_runtime_path(fingerprint: str, version: str, n=None, allow_existing=True) -> Path:
    """Returns ~/.brmspy/runtime/{fingerprint}-{version}/."""
    if n is None:
        runtime_path = get_runtime_base_dir() / f"{fingerprint}-{version}"
    else:
        runtime_path = get_runtime_base_dir() / f"{fingerprint}-{version}-{n}"
    if not allow_existing:
        if runtime_path.exists():
            if n is None:
                n = 0
            return get_runtime_path(fingerprint, version, n + 1)

    return runtime_path


def is_runtime_dir(path: Path) -> bool:
    """Check if path has valid structure (manifest.json, Rlib/, cmdstan/)."""
    if not path.is_dir():
        return False
    manifest = path / "manifest.json"
    return manifest.is_file()


def list_installed_runtimes() -> list[Path]:
    """List all installed runtime directories."""
    base_dir = get_runtime_base_dir()
    if not base_dir.exists():
        return []

    runtimes = []
    for item in base_dir.iterdir():
        if item.is_dir() and is_runtime_dir(item):
            runtimes.append(item)
    return runtimes


def find_runtime_by_fingerprint(fingerprint: str) -> Path | None:
    """Find newest installed runtime matching fingerprint."""
    runtimes = list_installed_runtimes()
    matching = [r for r in runtimes if fingerprint in r.name]
    if not matching:
        return None
    # Sort by name (which includes version) and return newest
    return sorted(matching, reverse=True)[0]


def read_stored_hash(path: Path) -> str | None:
    """Read hash file from runtime directory."""
    hash_path = path / "hash"
    if not hash_path.is_file():
        return None
    return hash_path.read_text(encoding="utf-8").strip()


def write_stored_hash(path: Path, hash_value: str) -> None:
    """Write hash file to runtime directory."""
    hash_path = path / "hash"
    hash_path.write_text(hash_value.strip() + "\n", encoding="utf-8")


def install_from_archive(
    archive: Path,
    fingerprint: str,
    version: str,
) -> Path:
    """
    Extract archive to runtime directory.
    Returns path to installed runtime.
    """
    import time

    base_dir = get_runtime_base_dir()
    is_windows = get_os() == "windows"
    runtime_root = get_runtime_path(fingerprint, version, allow_existing=not is_windows)

    # Extract to temp directory first
    temp_extract_root = base_dir / "_tmp_extract"
    if temp_extract_root.exists():
        shutil.rmtree(temp_extract_root)
    temp_extract_root.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive, mode="r:*") as tf:
            # 'data' filter breaks on windows and python 3.14 demands a filter.
            # fully_trusted is least error prone for now
            tf.extractall(path=temp_extract_root, filter="fully_trusted")

        # Find the runtime directory in extracted content
        runtime_tmp = temp_extract_root / "runtime"
        if not runtime_tmp.is_dir():
            raise RuntimeError(
                f"Extracted archive does not contain 'runtime/' under {temp_extract_root}"
            )

        # Validate manifest
        manifest_path = runtime_tmp / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing manifest.json in {runtime_tmp}")

        # Remove existing runtime if present - critical for Windows!
        # On Windows, if runtime_root exists and rmtree fails silently,
        # shutil.move() will place runtime_tmp INSIDE runtime_root instead of replacing it.
        if runtime_root.exists():
            shutil.rmtree(runtime_root, ignore_errors=False)
            # Wait for deletion to complete on Windows (file locking issues)
            for _ in range(10):
                if not runtime_root.exists():
                    break
                time.sleep(0.1)

        # Move to final location
        shutil.move(str(runtime_tmp), str(runtime_root))

    finally:
        shutil.rmtree(temp_extract_root, ignore_errors=True)

    return runtime_root


def install_from_directory(
    source: Path,
    fingerprint: str,
    version: str,
) -> Path:
    """
    Copy/move directory to runtime location.
    Returns path to installed runtime.
    """
    runtime_root = get_runtime_path(fingerprint, version)

    # Validate manifest
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing manifest.json in {source}")

    source = source.resolve()
    runtime_root = runtime_root.resolve()

    # If already in place, just return
    if source == runtime_root:
        return runtime_root

    # Remove existing runtime if present
    if runtime_root.exists():
        shutil.rmtree(runtime_root, ignore_errors=True)

    # Move to final location
    shutil.move(str(source), str(runtime_root))

    return runtime_root


def remove_runtime(path: Path) -> None:
    """Remove installed runtime directory."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
