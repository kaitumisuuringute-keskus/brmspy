"""
Type definitions for runtime management.

All types are immutable (frozen dataclasses) to prevent accidental mutations.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemInfo:
    """Immutable system environment snapshot."""
    os: str                                    # 'linux', 'macos', 'windows'
    arch: str                                  # 'x86_64', 'arm64'
    r_version: tuple[int, int, int] | None     # (4, 3, 2)
    fingerprint: str                           # 'linux-x86_64-r4.3'
    
    # Toolchain (populated based on OS)
    glibc_version: tuple[int, int] | None      # Linux
    clang_version: tuple[int, int] | None      # macOS
    gxx_version: tuple[int, int] | None        # All platforms
    has_rtools: bool                           # Windows


@dataclass(frozen=True)
class RuntimeManifest:
    """Runtime bundle manifest (from manifest.json)."""
    runtime_version: str
    fingerprint: str
    r_version: str
    cmdstan_version: str
    r_packages: dict[str, str]    # {package_name: version}
    manifest_hash: str
    built_at: str


@dataclass(frozen=True)
class RuntimeStatus:
    """Immutable snapshot of current runtime state."""
    # Active state
    active_runtime: Path | None
    is_activated: bool                         # True if R env currently modified
    
    # System
    system: SystemInfo
    
    # Compatibility
    can_use_prebuilt: bool
    prebuilt_available: bool                   # For this fingerprint
    compatibility_issues: tuple[str, ...]      # Why prebuilt unavailable
    
    # Installed runtimes
    installed_runtimes: tuple[Path, ...]
    
    # Current R package versions
    brms_version: str | None
    cmdstanr_version: str | None
    rstan_version: str | None


@dataclass
class StoredEnv:
    """Captured R environment for restoration on deactivate."""
    lib_paths: list[str]
    cmdstan_path: str | None