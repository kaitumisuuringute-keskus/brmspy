"""
Installation orchestration for both traditional and prebuilt modes.
"""

import tempfile
from typing import List
import urllib.request
from pathlib import Path
from brmspy.runtime import (
    _platform, _r_packages, _r_env, _rtools,
    _github, _storage, _state
)


def install_traditional(
    *,
    brms_version: str | None = None,
    cmdstanr_version: str | None = None,
    install_rstan: bool = True,
    install_rtools: bool = False,
    install_cmdstanr: bool = True,
    rstan_version: str | None = None
) -> None:
    """
    Install brms via traditional R package installation.
    
    Installs into system R library, builds CmdStan from source.
    Takes 20-30 minutes typically.
    """
    # Validate
    _platform.require_r_available()
    
    # Setup
    _r_env.forward_github_token()
    
    if install_rtools and _platform.get_os() == "windows":
        _rtools.ensure_installed()
    
    repos_cmdstanr: List[str] = [
        'https://stan-dev.r-universe.dev',
        'https://mc-stan.org/r-packages/'
    ]

    # Install packages
    _r_packages.install_package("brms", version=brms_version, repos_extra=repos_cmdstanr)
    _r_packages.install_package_deps("brms", repos_extra=repos_cmdstanr)
    #_r_packages.install_package("StanHeaders", repos_extra=repos_cmdstanr)
    
    
    if install_cmdstanr:
        _r_packages.install_package("cmdstanr", version=cmdstanr_version, repos_extra=repos_cmdstanr)
        _r_packages.install_package_deps("cmdstanr", repos_extra=repos_cmdstanr)
        _r_packages.build_cmdstan()
    
    if install_rstan:
        _r_packages.install_package("rstan", version=rstan_version, repos_extra=repos_cmdstanr)
        _r_packages.install_package_deps("rstan", repos_extra=repos_cmdstanr)
    
    _state.invalidate_packages()
    _state.get_brms()


def install_runtime(
    *,
    install_rtools: bool = False,
) -> Path:
    """
    Install prebuilt runtime bundle.
    
    Downloads from GitHub, extracts to ~/.brmspy/runtime/.
    Does NOT activate - caller handles that.
    
    Returns:
        Path to installed runtime directory.
    """
    # Validate system compatibility
    _platform.require_prebuilt_compatible()
    
    # Setup
    _r_env.forward_github_token()
    
    if install_rtools and _platform.get_os() == "windows":
        _rtools.ensure_installed()
    
    fingerprint = _platform.system_fingerprint()
    version = _github.get_latest_runtime_version()
    
    # Check if already installed with matching hash
    existing = _storage.find_runtime_by_fingerprint(fingerprint)
    if existing:
        url = _github.get_runtime_download_url(fingerprint, version)
        expected_hash = _github.get_asset_sha256(url)
        stored_hash = _storage.read_stored_hash(existing)
        
        if expected_hash and stored_hash == expected_hash:
            return existing  # Reuse existing
    
    # Download and install
    url = _github.get_runtime_download_url(fingerprint, version)
    expected_hash = _github.get_asset_sha256(url)
    
    # Download to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download archive
        archive_path = temp_path / "runtime.tar.gz"
        urllib.request.urlretrieve(url, archive_path)
        
        # Install from archive
        runtime_path = _storage.install_from_archive(archive_path, fingerprint, version)
    
    if expected_hash:
        _storage.write_stored_hash(runtime_path, expected_hash)
    
    _state.invalidate_packages()
    
    return runtime_path