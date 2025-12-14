"""
Windows Rtools management. Split into focused functions.
"""

import os
import platform
import re
import string
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.error import ContentTooShortError, HTTPError, URLError

from packaging.version import Version

from brmspy._runtime._platform import get_arch, get_os
from brmspy.helpers.log import log_warning

# R version range -> Rtools version
RTOOLS_VERSIONS = {
    (4, 0): "40",
    (4, 2): "42",
    (4, 3): "43",
    (4, 4): "44",
    (4, 5): "45",
    (4, 6): "46",
    (4, 7): "47",
    (4, 8): "48",
}
RTOOLS_SUBDIRS = [
    os.path.join("usr", "bin"),
    os.path.join("mingw64", "bin"),
]


def _windows_drives() -> list[str]:
    """Return a list of existing drive roots like ['C:\\', 'D:\\', ...]."""
    drives = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if Path(root).exists():
            drives.append(root)
    return drives


def _candidate_rtools_paths() -> list[str]:
    """Generate all plausible Rtools bin paths across all drives."""
    candidates: list[str] = []
    for drive in _windows_drives():
        for ver in RTOOLS_VERSIONS.values():
            base = os.path.join(drive, f"rtools{ver}")
            base2 = os.path.join(drive, f"Rtools{ver}")
            for sub in RTOOLS_SUBDIRS:
                candidates.append(os.path.join(base, sub))
                candidates.append(os.path.join(base2, sub))
    return candidates


def get_required_version(r_version: tuple[int, int, int] | Version) -> str | None:
    """Map R version to required Rtools version."""
    if isinstance(r_version, Version):
        major, minor = r_version.major, r_version.minor
    else:
        major, minor, _ = r_version

    # Find the appropriate Rtools version
    for (r_major, r_minor), rtools_ver in sorted(RTOOLS_VERSIONS.items(), reverse=True):
        if major > r_major or (major == r_major and minor >= r_minor):
            return rtools_ver

    return None


RTOOLS_FALLBACK_URLS = {
    "40": "https://cran.r-project.org/bin/windows/Rtools/rtools40-x86_64.exe",
    "42": "https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5355-5357.exe",
    "43": "https://cran.r-project.org/bin/windows/Rtools/rtools43/files/rtools43-5976-5975.exe",
    "44": "https://cran.r-project.org/bin/windows/Rtools/rtools44/files/rtools44-6459-6401.exe",
    "45": "https://cran.r-project.org/bin/windows/Rtools/rtools45/files/rtools45-6691-6492.exe",
}

RTOOLS_BASE = "https://cran.r-project.org/bin/windows/Rtools"


def _discover_rtools_installer(
    rtools_version: str,
    timeout: float = 10.0,
    aarch64: bool = False,
) -> str | None:
    """
    Try to discover the latest Rtools installer .exe from the CRAN directory index.

    Looks at:
        https://cran.r-project.org/bin/windows/Rtools/rtools{version}/files/
    and picks the newest-looking `rtools{version}-*.exe`.

    If ``aarch64`` is True, prefer the ``-aarch64-`` installer.
    Otherwise prefer the x86_64 installer and avoid the aarch64 one.
    """
    index_url = f"{RTOOLS_BASE}/rtools{rtools_version}/files/"

    try:
        with urllib.request.urlopen(index_url, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError):
        return None

    # Match things like:
    #   rtools45-6691-6492.exe
    #   rtools45-aarch64-6691-6492.exe
    pattern = rf'href="(rtools{re.escape(rtools_version)}-[^"]+\.exe)"'
    matches = re.findall(pattern, html)
    if not matches:
        return None

    # Split by arch:
    aarch64_candidates = [m for m in matches if "-aarch64-" in m]
    x86_candidates = [m for m in matches if "-aarch64-" not in m]

    if aarch64:
        candidates = aarch64_candidates or x86_candidates
    else:
        candidates = x86_candidates or aarch64_candidates

    if not candidates:
        return None

    # Lexicographically last is usually the newest build
    filename = sorted(candidates)[-1]
    return index_url + filename


def get_download_url(rtools_version: str) -> str:
    """Get download URL for Rtools version."""
    # Try to dynamically discover from CRAN directory listing
    is_arm64 = get_arch() == "arm64"

    url = _discover_rtools_installer(rtools_version, aarch64=is_arm64)
    if url is not None:
        return url

    # Fall back to old hard-coded mapping if discovery fails
    if rtools_version in RTOOLS_FALLBACK_URLS:
        return RTOOLS_FALLBACK_URLS[rtools_version]

    # Probably will NOT work, but return it anyways
    return f"https://cran.r-project.org/bin/windows/Rtools/rtools{rtools_version}/files/rtools{rtools_version}-x86_64.exe"


def is_installed() -> bool:
    """Check if Rtools is installed (make + mingw g++ on PATH)."""
    try:
        # Check for make
        subprocess.run(
            ["make", "--version"], capture_output=True, check=True, timeout=10
        )

        # Check for mingw g++
        result = subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, check=True, timeout=10
        )

        # Verify it's mingw
        output = result.stdout.lower()
        if "mingw" in output or "rtools" in output:
            return True

        return False
    except Exception:
        return False


def get_installed_gxx_version() -> tuple[int, int] | None:
    """Get g++ version from Rtools."""
    try:
        result = subprocess.check_output(["g++", "--version"], text=True, timeout=10)
        # Parse version
        for line in result.splitlines():
            for token in line.split():
                if token[0].isdigit() and "." in token:
                    parts = token.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def _stream_download(url: str, dst: Path, timeout: float | None = 30) -> None:
    """Download URL to dst in chunks, verifying size if Content-Length is present."""
    CHUNK_SIZE = 1024 * 1024  # 1 MB

    with urllib.request.urlopen(url, timeout=timeout) as resp, dst.open("wb") as f:
        content_length = resp.headers.get("Content-Length")
        expected_size: int | None = int(content_length) if content_length else None

        total = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)

    if expected_size is not None and total != expected_size:
        raise OSError(
            f"incomplete download: got {total} bytes, expected {expected_size}"
        )


def download_installer(rtools_version: str, max_retries: int = 3) -> Path:
    """Download Rtools installer to temp directory with retries and size check."""
    url = get_download_url(rtools_version)
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _stream_download(url, tmp_path)
            # If we got here, download is complete (or server didn't send length).
            return tmp_path

        except (OSError, URLError, ContentTooShortError) as e:
            last_err = e
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

            log_warning(
                f"[rtools] download attempt {attempt}/{max_retries} failed: {e}"
            )

    raise RuntimeError(
        f"Failed to download Rtools installer from {url} after {max_retries} attempts"
    ) from last_err


def run_installer(installer: Path, rtools_version: str, silent: bool = True) -> None:
    """Run Rtools installer safely on Windows and CI."""
    system_drive = os.environ.get("SystemDrive", "C:")
    assert rtools_version in list(RTOOLS_VERSIONS.values())

    install_dir = Path(system_drive) / f"Rtools{rtools_version}"

    # Ensure parent directory exists to avoid ACL weirdness
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    args = [str(installer)]

    if silent:
        args.extend(
            [
                "/SP-",  # skip intro dialog (CRITICAL)
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                f"/DIR={install_dir}",  # no inner quotes
            ]
        )

    # 5 minute timeout - Rtools installers typically complete in 1-3 minutes
    subprocess.run(args, check=True, timeout=300)


def update_paths() -> None:
    """Update PATH in both Python os.environ and R Sys.setenv."""
    if get_os() != "windows":
        return None
    import rpy2.robjects as ro

    current_path = os.environ.get("PATH", "")
    current_entries = current_path.split(os.pathsep) if current_path else []

    new_entries: list[str] = []

    for candidate in _candidate_rtools_paths():
        p = Path(candidate)
        if p.exists():
            # avoid duplicates in both new_entries and existing PATH
            if candidate not in current_entries and candidate not in new_entries:
                new_entries.append(candidate)

    if not new_entries:
        return

    # Update Python PATH
    os.environ["PATH"] = os.pathsep.join(new_entries + current_entries)

    # Update R PATH using rpy2 in a safe way (no manual quoting)
    try:
        sys_setenv = cast(Callable, ro.r("Sys.setenv"))
        sys_setenv(PATH=os.environ["PATH"])
    except Exception:
        # Best-effort; don't crash if R isn't ready
        pass


def ensure_installed() -> None:
    """
    Orchestrator: ensure Rtools is installed for current R.
    Downloads and installs if needed. Updates paths.
    """
    if platform.system() != "Windows":
        return

    # Check if already installed
    if is_installed():
        update_paths()
        return

    # Get R version and determine required Rtools version
    from brmspy._runtime._platform import get_r_version

    r_ver = get_r_version()
    if r_ver is None:
        raise RuntimeError("Cannot determine R version")

    rtools_ver = get_required_version(r_ver)
    if rtools_ver is None:
        raise RuntimeError(f"No Rtools version available for R {r_ver}")

    # Download and install
    installer = download_installer(rtools_ver)
    try:
        run_installer(installer, rtools_version=rtools_ver, silent=True)
    finally:
        if installer.exists():
            installer.unlink()

    # Update paths
    update_paths()

    # Verify installation
    if not is_installed():
        raise RuntimeError("Rtools installation failed")
