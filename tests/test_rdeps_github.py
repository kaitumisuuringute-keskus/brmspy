"""
Tests for brmspy.runtime._github module (GitHub API interactions).

Focus: URL parsing and GitHub release operations.

Note: These tests use mocks for error scenarios that cannot be reliably
triggered with real GitHub API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from brmspy.runtime._github import (
    parse_release_url,
    fetch_release_metadata,
    get_asset_sha256,
    get_github_asset_sha256_from_url,
    get_runtime_download_url,
)


@pytest.mark.rdeps
class TestGitHubReleaseURL:
    """Test GitHub release URL parsing."""
    
    def test_parse_invalid_url_format(self):
        """Test that invalid GitHub URL format raises ValueError."""
        invalid_urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo/releases",
            "https://example.com/file.tar.gz",
            "https://github.com/owner/repo/issues/123",
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Malformed GitHub release URL"):
                parse_release_url(url)
    
    def test_parse_valid_url(self):
        """Test parsing valid GitHub release URL."""
        url = "https://github.com/owner/repo/releases/download/v1.0.0/asset.tar.gz"
        owner, repo, tag, asset_name = parse_release_url(url)
        
        assert owner == "owner"
        assert repo == "repo"
        assert tag == "v1.0.0"
        assert asset_name == "asset.tar.gz"


@pytest.mark.rdeps
class TestGitHubAPIFetch:
    """Test GitHub API metadata fetching."""
    
    def test_fetch_release_metadata_success(self):
        """Test successful release metadata fetch."""
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.read.return_value = b'{"tag_name": "v1.0.0", "assets": []}'
        
        with patch('urllib.request.urlopen', return_value=mock_response):
            result = fetch_release_metadata("owner", "repo", "v1.0.0")
            assert result == {"tag_name": "v1.0.0", "assets": []}
    
    def test_fetch_release_metadata_with_auth_token(self):
        """Test that GITHUB_TOKEN is used when available."""
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.read.return_value = b'{"tag_name": "v1.0.0"}'
        
        with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
            with patch.dict('os.environ', {'GITHUB_TOKEN': 'test_token'}):
                fetch_release_metadata("owner", "repo", "v1.0.0")
                
                # Verify auth header was added
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.headers.get('Authorization') == 'token test_token'
    
    def test_fetch_release_metadata_connection_error(self):
        """Test handling of connection errors."""
        with patch('urllib.request.urlopen', side_effect=Exception("Network error")):
            with pytest.raises(ConnectionError, match="Failed to fetch release metadata"):
                fetch_release_metadata("owner", "repo", "v1.0.0")


@pytest.mark.rdeps
class TestAssetSHA256:
    """Test SHA256 retrieval from GitHub assets."""
    
    def test_get_asset_sha256_returns_none_currently(self):
        """Test that get_asset_sha256 returns None (not yet implemented)."""
        url = "https://github.com/owner/repo/releases/download/v1.0.0/asset.tar.gz"
        
        mock_release_data = {
            "assets": [
                {"name": "asset.tar.gz", "id": 123},
            ]
        }
        
        with patch('brmspy.runtime._github.fetch_release_metadata', return_value=mock_release_data):
            result = get_asset_sha256(url)
            assert result is None
    
    def test_get_github_asset_sha256_from_url_without_require(self):
        """Test wrapper function without require_digest."""
        url = "https://github.com/owner/repo/releases/download/v1.0.0/asset.tar.gz"
        
        with patch('brmspy.runtime._github.get_asset_sha256', return_value=None):
            result = get_github_asset_sha256_from_url(url, require_digest=False)
            assert result is None
    
    def test_get_github_asset_sha256_from_url_with_require_digest(self):
        """Test error when digest required but not available."""
        url = "https://github.com/owner/repo/releases/download/v1.0.0/asset.tar.gz"
        
        with patch('brmspy.runtime._github.get_asset_sha256', return_value=None):
            with pytest.raises(ValueError, match="Could not fetch SHA256 digest"):
                get_github_asset_sha256_from_url(url, require_digest=True)


@pytest.mark.rdeps
class TestRuntimeDownloadURL:
    """Test runtime download URL construction."""
    
    def test_get_runtime_download_url_with_version(self):
        """Test URL construction with specific version."""
        url = get_runtime_download_url("linux-x86_64-r4.3", version="1.2.3")
        
        from urllib.parse import urlparse
        assert urlparse(url).netloc == "github.com"
        assert "brmspy-runtime-1.2.3-linux-x86_64-r4.3.tar.gz" in url
    
    def test_get_runtime_download_url_with_latest(self):
        """Test URL construction with 'latest' version."""
        with patch('brmspy.runtime._github.get_latest_runtime_version', return_value="0.1.0"):
            url = get_runtime_download_url("macos-arm64-r4.4", version="latest")
            
            assert "brmspy-runtime-0.1.0-macos-arm64-r4.4.tar.gz" in url