import os
import platform
import subprocess
import tempfile
from typing import Optional, cast
from urllib import request
from packaging.version import Version
from rpy2 import robjects as ro

def _get_r_version() -> Version:
    """Return R version as packaging.Version."""
    ver_str = str(cast(list, ro.r('as.character(getRversion())'))[0])
    return Version(ver_str)


def pick_rtools_for_r(r_ver: Version) -> str | None:
    """
    Return the Rtools 'major' tag: '40', '42', '43', '44', '45', or None for legacy.
    """
    if r_ver < Version("4.0.0"):
        return None  # legacy, not worth supporting in brmspy
    if r_ver < Version("4.2.0"):
        return "40"  # R 4.0.x–4.1.x
    if r_ver < Version("4.3.0"):
        return "42"  # R 4.2.x
    if r_ver < Version("4.4.0"):
        return "43"  # R 4.3.x
    if r_ver < Version("4.5.0"):
        return "44"  # R 4.4.x
    return "45"      # R 4.5.x and up


RTOOLS_INSTALLERS = {
    "40": "https://cran.r-project.org/bin/windows/Rtools/rtools40-x86_64.exe",
    "42": "https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5355-5357.exe",
    "43": "https://cran.r-project.org/bin/windows/Rtools/rtools43/files/rtools43-5976-5975.exe",
    "44": "https://cran.r-project.org/bin/windows/Rtools/rtools44/files/rtools44-6459-6401.exe",
    "45": "https://cran.r-project.org/bin/windows/Rtools/rtools45/files/rtools45-6691-6492.exe",
}

def _silent_install_exe(url: str, label: str) -> None:
    tmp_dir = tempfile.gettempdir()
    exe_path = os.path.join(tmp_dir, f"{label}.exe")
    print(f"brmspy: downloading {label} from {url}")
    request.urlretrieve(url, exe_path)
    print(f"brmspy: running {label} installer silently...")
    subprocess.run(
        [
            exe_path,
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
        ],
        check=True,
    )


def _install_rtools_for_current_r(ci_only: bool = True) -> Optional[str]:
    """
    Detect R version via rpy2, install matching Rtools if missing.
    Returns the rtools tag ('40', '42', ...) or None.
    """
    if platform.system() != "Windows":
        return None

    if ci_only and os.environ.get("GITHUB_ACTIONS") != "true":
        print("brmspy: not in CI, refusing to auto-install Rtools.")
        return None

    r_ver = _get_r_version()
    tag = pick_rtools_for_r(r_ver)
    if tag is None:
        print(f"brmspy: R {r_ver} is too old for automatic Rtools handling.")
        return None

    url = RTOOLS_INSTALLERS.get(tag)
    if not url:
        print(f"brmspy: no installer URL configured for Rtools{tag}")
        return None

    _silent_install_exe(url, f"rtools{tag}")

    # Best-effort PATH tweaks for current process (default layout)
    root = rf"C:\rtools{tag}"
    candidates = [
        os.path.join(root, "usr", "bin"),
        os.path.join(root, "x86_64-w64-mingw32.static.posix", "bin"),
        os.path.join(root, "mingw64", "bin"),  # varies by version
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            print(f"brmspy: added to PATH: {p}")

    return tag