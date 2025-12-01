from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from brmspy.helpers.log import log_warning


@dataclass
class GitHubReleaseAssetMetadata:
    owner: str
    repo: str
    tag: str
    name: str
    id: int
    browser_download_url: str
    digest_raw: Optional[str]        # e.g. "sha256:abcd..."
    digest_algorithm: Optional[str]  # e.g. "sha256"
    digest_value: Optional[str]      # e.g. "abcd..."
    size: int


def _parse_github_release_download_url(url: str) -> Tuple[str, str, str, str]:
    """
    Parse a URL like:
      https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}

    Returns (owner, repo, tag, asset_name).
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    # owner / repo / releases / download / tag / asset...
    if len(parts) < 6 or parts[2] != "releases" or parts[3] != "download":
        raise ValueError(f"Not a GitHub release download URL: {url!r}")

    owner, repo, _, _, tag = parts[:5]
    asset_name = "/".join(parts[5:])
    return owner, repo, tag, asset_name


def _github_get_json(api_url: str, token: Optional[str] = None) -> dict:
    """
    GET JSON from GitHub API.

    - If `token` is None, look at GITHUB_TOKEN / GH_TOKEN in env.
    - If an auth token is used and returns 401/403, retry once
      without Authorization (public repo fallback).
    """
    env_token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

    base_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "brmspy-runtime-installer",
    }

    def request(use_auth: bool) -> dict:
        headers = dict(base_headers)
        if use_auth and env_token:
            headers["Authorization"] = f"Bearer {env_token}"
        req = Request(api_url, headers=headers, method="GET")
        with urlopen(req, timeout=30) as resp:
            return json.load(resp)

    if env_token:
        try:
            return request(use_auth=True)
        except HTTPError as e:
            if e.code in (401, 403):
                # Invalid / insufficient token – retry anonymously.
                log_warning("[github] Auth token rejected, retrying without credentials")
                return request(use_auth=False)
            raise
    else:
        return request(use_auth=False)


def get_github_release_asset_metadata_from_url(
    url: str,
    token: Optional[str] = None,
    require_digest: bool = False,
) -> GitHubReleaseAssetMetadata:
    """
    Resolve a GitHub release download URL to API metadata and digest.

    Parameters
    ----------
    url : str
        https://github.com/OWNER/REPO/releases/download/TAG/ASSET_NAME
    token : str, optional
        Explicit token. If None, uses $GITHUB_TOKEN or $GH_TOKEN if set.
    require_digest : bool, default False
        If True, raise if the asset has no `digest` field.

    Returns
    -------
    GitHubReleaseAssetMetadata
    """
    owner, repo, tag, asset_name = _parse_github_release_download_url(url)

    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    )
    release = _github_get_json(api_url, token=token)

    assets = release.get("assets") or []
    asset = next((a for a in assets if a.get("name") == asset_name), None)
    if asset is None:
        raise ValueError(
            f"Asset {asset_name!r} not found in release tag {tag!r} "
            f"for {owner}/{repo}"
        )

    digest_raw = asset.get("digest")  # e.g. "sha256:abcd..."
    algo = value = None
    if digest_raw and ":" in digest_raw:
        algo, _, value = digest_raw.partition(":")

    if require_digest and not value:
        raise RuntimeError(
            f"Asset {asset_name!r} in {owner}/{repo}@{tag} has no digest metadata"
        )

    return GitHubReleaseAssetMetadata(
        owner=owner,
        repo=repo,
        tag=tag,
        name=asset_name,
        id=asset["id"],
        browser_download_url=asset["browser_download_url"],
        digest_raw=digest_raw,
        digest_algorithm=algo,
        digest_value=value,
        size=asset.get("size", 0),
    )


def get_github_asset_sha256_from_url(
    url: str,
    token: Optional[str] = None,
    require_digest: bool = True,
) -> Optional[str]:
    """
    Convenience: return bare SHA-256 hex string for an asset URL.

    Returns None if no digest present and `require_digest` is False.
    """
    meta = get_github_release_asset_metadata_from_url(
        url=url,
        token=token,
        require_digest=require_digest,
    )
    return meta.digest_value
