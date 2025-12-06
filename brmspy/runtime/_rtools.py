"""
Windows Rtools management. Split into focused functions.
"""

import os
import platform
import re
import subprocess
import tempfile
from urllib.error import ContentTooShortError, HTTPError, URLError
import urllib.request
from pathlib import Path
from typing import Optional, cast

import rpy2.robjects as ro

from brmspy.helpers.log import log_warning
from packaging.version import Version


# R version range -> Rtools version
RTOOLS_VERSIONS = {
    (4, 0): "40",
    (4, 2): "42",
    (4, 3): "43",
    (4, 4): "44",
    (4, 5): "45",
    (4, 6): "46",
    (4, 7): "47"
}


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
    x86_candidates     = [m for m in matches if "-aarch64-" not in m]

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
    url = _discover_rtools_installer(rtools_version)
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
        subprocess.run(["make", "--version"], 
                      capture_output=True, check=True)
        
        # Check for mingw g++
        result = subprocess.run(["g++", "--version"],
                               capture_output=True, text=True, check=True)
        
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
        result = subprocess.check_output(["g++", "--version"], text=True)
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


def _stream_download(url: str, dst: Path) -> None:
    """Download URL to dst in chunks, verifying size if Content-Length is present."""
    CHUNK_SIZE = 1024 * 1024  # 1 MB

    with urllib.request.urlopen(url) as resp, dst.open("wb") as f:
        content_length = resp.headers.get("Content-Length")
        expected_size: Optional[int] = int(content_length) if content_length else None

        total = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)

    if expected_size is not None and total != expected_size:
        raise IOError(
            f"incomplete download: got {total} bytes, expected {expected_size}"
        )


def download_installer(rtools_version: str, max_retries: int = 3) -> Path:
    """Download Rtools installer to temp directory with retries and size check."""
    url = get_download_url(rtools_version)
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _stream_download(url, tmp_path)
            # If we got here, download is complete (or server didn't send length).
            return tmp_path

        except (URLError, IOError, ContentTooShortError) as e:
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




def run_installer(installer: Path, silent: bool = True) -> None:
    """Run Rtools installer."""
    args = [str(installer)]
    if silent:
        args.extend(["/VERYSILENT", "/SUPPRESSMSGBOXES"])
    
    subprocess.run(args, check=True)


def update_paths() -> None:
    """Update PATH in both Python os.environ and R Sys.setenv."""
    # Update Python PATH
    rtools_paths = [
        r"C:\rtools48\usr\bin",
        r"C:\rtools48\mingw64\bin",
        r"C:\rtools47\usr\bin",
        r"C:\rtools47\mingw64\bin",
        r"C:\rtools46\usr\bin",
        r"C:\rtools46\mingw64\bin",
        r"C:\rtools45\usr\bin",
        r"C:\rtools45\mingw64\bin",
        r"C:\rtools44\usr\bin",
        r"C:\rtools44\mingw64\bin",
        r"C:\rtools43\usr\bin",
        r"C:\rtools43\mingw64\bin",
        r"C:\rtools42\usr\bin",
        r"C:\rtools42\mingw64\bin",
        r"C:\rtools40\usr\bin",
        r"C:\rtools40\mingw64\bin",
    ]
    
    current_path = os.environ.get("PATH", "")
    new_paths = []
    
    for rtools_path in rtools_paths:
        if Path(rtools_path).exists() and rtools_path not in current_path:
            new_paths.append(rtools_path)
    
    if new_paths:
        os.environ["PATH"] = os.pathsep.join(new_paths) + os.pathsep + current_path
        
        # Also update R's PATH
        try:
            new_path_str = os.pathsep.join(new_paths + [current_path])
            ro.r(f'Sys.setenv(PATH = "{new_path_str}")')
        except Exception:
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
    from brmspy.runtime._platform import get_r_version
    r_ver = get_r_version()
    if r_ver is None:
        raise RuntimeError("Cannot determine R version")
    
    rtools_ver = get_required_version(r_ver)
    if rtools_ver is None:
        raise RuntimeError(f"No Rtools version available for R {r_ver}")
    
    # Download and install
    installer = download_installer(rtools_ver)
    try:
        run_installer(installer, silent=True)
    finally:
        if installer.exists():
            installer.unlink()
    
    # Update paths
    update_paths()
    
    # Verify installation
    if not is_installed():
        raise RuntimeError("Rtools installation failed")
    


def _windows_has_rtools(silent=False) -> bool:
    # If Rtools is already found, skip (cmdstanr check)
    try:
        # Check if 'make' is available. If yes, we probably have Rtools.
        make_path = str(cast(list, ro.r('Sys.which("make")'))[0])
        if make_path and "rtools" in make_path.lower():
            return True
    except Exception:
        pass

    try:
        out = subprocess.check_output(["g++", "--version"], text=True, shell=True)
    except Exception:
        if not silent:
            log_warning(f"g++ not found")
        return False

    # Very rough: we expect mingw in the banner
    if "mingw" not in out.lower():
        if not silent:
            log_warning(f"mingw not found in g++ banner")
        return False

    return True