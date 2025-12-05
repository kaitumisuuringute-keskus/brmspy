"""
Pack runtime into distributable archive.
"""

import tarfile
from pathlib import Path

from brmspy.helpers.log import log


def pack_runtime(runtime_root: Path, out_dir: Path, runtime_version: str) -> Path:
    """
    Create compressed tar archive from runtime directory.
    
    Packages the staged runtime directory into a distributable .tar.gz
    archive with standardized naming for platform/version identification.
    
    Archive naming format:
    brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz
    
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
    """
    fingerprint = runtime_root.name
    archive_name = f"brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz"
    archive_path = out_dir / archive_name

    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[tar] Creating archive {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tf:
        # Add the runtime root directory contents under "runtime/"
        tf.add(runtime_root, arcname="runtime")

    return archive_path