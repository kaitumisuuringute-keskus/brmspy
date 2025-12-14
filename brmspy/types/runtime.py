"""
Runtime management types.

These dataclasses are returned by runtime/status helper functions and are used
internally to decide whether a "prebuilt runtime" can be used on the current
machine.

Conceptually:

- A *runtime* is a bundle containing CmdStan and a set of R packages installed
  into an isolated library directory (typically under `~/.brmspy/runtime/...`).
- An *environment* (see [`EnvironmentConfig`][brmspy.types.session.EnvironmentConfig]) is a
  named, user-managed library (`~/.brmspy/environment/<name>/Rlib`) layered on
  top of a runtime.

Notes
-----
All snapshot types in this module are immutable (`frozen=True`) so they can be
cached safely and won't be mutated accidentally by callers.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemInfo:
    """
    Immutable snapshot of the system environment relevant to runtime selection.

    Attributes
    ----------
    os : str
        Operating system identifier (for example ``"linux"``, ``"macos"``, ``"windows"``).
    arch : str
        CPU architecture (for example ``"x86_64"``, ``"arm64"``).
    r_version : tuple[int, int, int] or None
        Detected R version as ``(major, minor, patch)``.
    fingerprint : str
        Fingerprint used for runtime lookup and caching (for example ``"linux-x86_64-r4.3"``).

    glibc_version : tuple[int, int] or None
        Linux-only: glibc major/minor version used for compatibility checks.
    clang_version : tuple[int, int] or None
        macOS-only: clang major/minor version.
    gxx_version : tuple[int, int] or None
        C++ compiler major/minor version if detected.
    has_rtools : bool
        Windows-only: whether Rtools is available.
    """

    os: str  # 'linux', 'macos', 'windows'
    arch: str  # 'x86_64', 'arm64'
    r_version: tuple[int, int, int] | None  # (4, 3, 2)
    fingerprint: str  # 'linux-x86_64-r4.3'

    # Toolchain (populated based on OS)
    glibc_version: tuple[int, int] | None  # Linux
    clang_version: tuple[int, int] | None  # macOS
    gxx_version: tuple[int, int] | None  # All platforms
    has_rtools: bool  # Windows


@dataclass(frozen=True)
class RuntimeManifest:
    """
    Manifest for a prebuilt runtime bundle.

    This structure is typically loaded from a `manifest.json` stored alongside a
    runtime directory.

    Attributes
    ----------
    runtime_version : str
        brmspy runtime bundle version.
    fingerprint : str
        System fingerprint this runtime was built for.
    r_version : str
        R version string used for the runtime build (for example ``"4.5.0"``).
    cmdstan_version : str
        CmdStan version included in the runtime.
    r_packages : dict[str, str]
        Mapping of R package names to versions.
    manifest_hash : str
        Hash used to validate the runtime contents.
    built_at : str
        Build timestamp.
    """

    runtime_version: str
    fingerprint: str
    r_version: str
    cmdstan_version: str
    r_packages: dict[str, str]  # {package_name: version}
    manifest_hash: str
    built_at: str


@dataclass(frozen=True)
class RuntimeStatus:
    """
    Immutable snapshot of current runtime state.

    Attributes
    ----------
    active_runtime : pathlib.Path or None
        Path to the currently selected runtime (if any).
    is_activated : bool
        Whether the worker's embedded R session has been modified/activated to
        use the active runtime.
    system : SystemInfo
        Detected system info used for compatibility evaluation.
    can_use_prebuilt : bool
        Whether a prebuilt runtime could be used in principle.
    prebuilt_available : bool
        Whether a compatible prebuilt runtime is available for the current fingerprint.
    compatibility_issues : tuple[str, ...]
        Human-readable reasons why prebuilt runtime usage is not possible.
    installed_runtimes : tuple[pathlib.Path, ...]
        Runtime directories currently installed under brmspy storage.
    brms_version, cmdstanr_version, rstan_version : str or None
        Detected versions in the current worker session (if available).
    """

    # Active state
    active_runtime: Path | None
    is_activated: bool  # True if R env currently modified

    # System
    system: SystemInfo

    # Compatibility
    can_use_prebuilt: bool
    prebuilt_available: bool  # For this fingerprint
    compatibility_issues: tuple[str, ...]  # Why prebuilt unavailable

    # Installed runtimes
    installed_runtimes: tuple[Path, ...]

    # Current R package versions
    brms_version: str | None
    cmdstanr_version: str | None
    rstan_version: str | None


@dataclass
class StoredEnv:
    """
    Captured R environment used for deactivation/restoration.

    Attributes
    ----------
    lib_paths : list[str]
        `.libPaths()` values captured before activation.
    cmdstan_path : str or None
        CmdStan path captured before activation.
    """

    lib_paths: list[str]
    cmdstan_path: str | None
