"""
Windows Rtools management. Split into focused functions.
"""

import os
import platform
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import rpy2.robjects as ro


# R version range -> Rtools version
RTOOLS_VERSIONS = {
    (4, 0): "40",
    (4, 2): "42",
    (4, 3): "43",
    (4, 4): "44",
    (4, 5): "45",
}


def get_required_version(r_version: tuple[int, int, int]) -> str | None:
    """Map R version to required Rtools version."""
    major, minor, _ = r_version
    
    # Find the appropriate Rtools version
    for (r_major, r_minor), rtools_ver in sorted(RTOOLS_VERSIONS.items(), reverse=True):
        if major > r_major or (major == r_major and minor >= r_minor):
            return rtools_ver
    
    return None


def get_download_url(rtools_version: str) -> str:
    """Get download URL for Rtools version."""
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


def download_installer(rtools_version: str) -> Path:
    """Download Rtools installer to temp directory."""
    url = get_download_url(rtools_version)
    
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    urllib.request.urlretrieve(url, tmp_path)
    return tmp_path


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