"""
Manifest parsing and validation. Pure functions.
"""

import json
import hashlib
from pathlib import Path
from brmspy.helpers.log import log_warning
from brmspy.runtime._types import RuntimeManifest


def parse_manifest(path: Path) -> RuntimeManifest | None:
    """Parse manifest.json. Returns None if missing/invalid."""
    if not path.exists():
        return None
    
    data = None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        return RuntimeManifest(
            runtime_version=data.get("runtime_version", ""),
            fingerprint=data.get("fingerprint", ""),
            r_version=data.get("r_version", ""),
            cmdstan_version=data.get("cmdstan_version", ""),
            r_packages=data.get("r_packages", {}),
            manifest_hash=data.get("manifest_hash", ""),
            built_at=data.get("built_at", ""),
        )
    except Exception as e:
        log_warning(f"Failed parsing manifest: {e}")
        log_warning(f"Broken manifest contents: {data}")
        return None


def validate_manifest(manifest: RuntimeManifest, expected_fingerprint: str) -> None:
    """
    Validate manifest matches expected fingerprint.
    Raises RuntimeError with details if mismatch.
    """
    if manifest.fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"Runtime fingerprint mismatch: "
            f"manifest={manifest.fingerprint}, expected={expected_fingerprint}"
        )


def compute_manifest_hash(manifest_dict: dict) -> str:
    """Compute SHA256 of manifest content."""
    # Create a copy without the hash field itself
    data = {k: v for k, v in manifest_dict.items() if k != "manifest_hash"}
    
    # Serialize to JSON in a deterministic way
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    
    # Compute SHA256
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()