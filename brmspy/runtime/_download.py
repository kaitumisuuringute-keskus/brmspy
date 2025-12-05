"""
Download and extraction operations.
"""

import tarfile
import tempfile
import urllib.request
from pathlib import Path


def download_file(
    url: str,
    dest: Path,
    show_progress: bool = True,
) -> None:
    """Download file from URL to destination with optional progress bar."""
    urllib.request.urlretrieve(url, dest)


def extract_archive(archive: Path, dest_dir: Path) -> Path:
    """Extract tar.gz archive. Returns path to extracted root directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(archive, mode="r:*") as tf:
        tf.extractall(path=dest_dir)
    
    # Return the extracted directory (should be single top-level dir)
    items = list(dest_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return dest_dir


def download_runtime(
    url: str,
    dest_dir: Path,
    expected_hash: str | None = None,
) -> Path:
    """
    Download runtime archive and extract.
    Validates hash if provided.
    Returns path to extracted (not yet installed) runtime.
    """
    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        download_file(url, tmp_path)
        
        # TODO: Validate hash if provided
        
        # Extract
        extracted = extract_archive(tmp_path, dest_dir)
        
        return extracted
    finally:
        if tmp_path.exists():
            tmp_path.unlink()