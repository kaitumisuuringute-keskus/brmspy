"""
GitHub API operations for runtime downloads.
"""

import functools
import json
import os
import urllib.request
from urllib.parse import urlparse

from brmspy.helpers.log import log_warning

REPO_OWNER = "kaitumisuuringute-keskus"
REPO_NAME = "brmspy"


def parse_release_url(url: str) -> tuple[str, str, str, str]:
    """
    Parse a GitHub release asset URL into (owner, repo, tag, asset_name).

    Expected pattern:
        https://github.com/<owner>/<repo>/releases/download/<tag>/<asset_name>
    """
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Malformed GitHub release URL: Invalid scheme: {url}")

    if parsed.hostname != "github.com":
        raise ValueError(
            f"Malformed GitHub release URL: Unexpected host: {parsed.hostname!r}"
        )

    # Split path, ignoring leading slash
    parts = parsed.path.lstrip("/").split("/")

    # Expected:
    #   0: owner
    #   1: repo
    #   2: releases
    #   3: download
    #   4: tag
    #   5: asset_name
    if len(parts) < 6:
        raise ValueError(f"Malformed GitHub release URL path: {parsed.path!r}")

    if parts[2] != "releases" or parts[3] != "download":
        raise ValueError(
            f"Malformed GitHub release URL: Path does not match releases/download structure: {parsed.path!r}"
        )

    owner, repo, _, _, tag, asset_name = parts[:6]
    return owner, repo, tag, asset_name


def fetch_release_metadata(owner: str, repo: str, tag: str, use_token=True) -> dict:
    """Fetch release metadata from GitHub API (handles auth + retries)."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"

    # Prepare headers with auth if available
    headers = {"Accept": "application/vnd.github.v3+json"}
    if use_token:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        if token:
            headers["Authorization"] = f"token {token}"

    req = urllib.request.Request(api_url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception as e:
        raise ConnectionError(f"Failed to fetch release metadata: {e}") from e


@functools.cache  # to avoid rate limits.
def get_asset_sha256(url: str) -> str | None:
    """Get SHA256 hash from release asset metadata."""
    try:
        owner, repo, tag, asset_name = parse_release_url(url)
        try:
            metadata = fetch_release_metadata(owner, repo, tag, use_token=False)
        except Exception as e:
            log_warning(
                f"Anonymous asset sha256 fetching failed, trying with token... Reason: {e}"
            )
            metadata = fetch_release_metadata(owner, repo, tag, use_token=True)

        # Find the asset in the release
        for asset in metadata.get("assets", []):
            if asset.get("name") == asset_name:
                # GitHub doesn't provide SHA256 directly, but we can check if there's
                # a .sha256 file or similar
                # For now, return None as this needs to be implemented based on
                # how the releases are structured
                return asset.get("digest")

        return None
    except Exception as e:
        log_warning(f"Could not get asset metadata: {e}")
        return None


def get_runtime_download_url(fingerprint: str, version: str = "latest") -> str:
    """Construct download URL for runtime bundle."""
    if version == "latest":
        version = get_latest_runtime_version()
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/runtime/brmspy-runtime-{version}-{fingerprint}.tar.gz"


def get_latest_runtime_version() -> str:
    """Query GitHub for latest runtime release version."""
    # For now, return a default version
    # This should query the GitHub API for the latest release
    return "0.2.0"


def get_github_asset_sha256_from_url(
    url: str, require_digest: bool = False
) -> str | None:
    """Get SHA256 from GitHub release asset. Used by old code."""
    sha = get_asset_sha256(url)
    if require_digest and sha is None:
        raise ValueError(f"Could not fetch SHA256 digest for {url}")
    return sha
