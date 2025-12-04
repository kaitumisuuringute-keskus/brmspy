"""
Tests for brmspy.binaries.github module (GitHub API interactions).

Focus: Edge cases and error handling for GitHub release asset metadata.

Note: These tests REQUIRE mocks because error scenarios (401/403, invalid URLs, 
missing assets, missing digests) cannot be reliably triggered with real GitHub API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from urllib.error import HTTPError
from email.message import Message

from brmspy.binaries.github import (
    _parse_github_release_download_url,
    _github_get_json,
    get_github_release_asset_metadata_from_url,
    get_github_asset_sha256_from_url,
)


@pytest.mark.crossplatform
class TestGitHubReleaseURL:
    """Test GitHub release URL parsing."""
    
    def test_parse_invalid_url_format(self):
        """Test that invalid GitHub URL format raises ValueError (line 40)."""
        invalid_urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo/releases",
            "https://example.com/file.tar.gz",
            "https://github.com/owner/repo/issues/123",
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Not a GitHub release download URL"):
                _parse_github_release_download_url(url)


@pytest.mark.crossplatform 
class TestGitHubAPIAuth:
    """Test GitHub API authentication and retry logic."""
    
    def test_api_auth_token_retry_on_401(self):
        """Test auth token retry logic on 401 Unauthorized (lines 79-81)."""
        error_401 = HTTPError(
            url="https://api.github.com/test",
            code=401,
            msg="Unauthorized",
            hdrs=Message(),
            fp=None
        )
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.read.return_value = b'{"public": "data"}'
        
        call_count = [0]
        
        def mock_urlopen(request, timeout=30):
            call_count[0] += 1
            if call_count[0] == 1:
                raise error_401
            return mock_response
        
        with patch('brmspy.binaries.github.urlopen', side_effect=mock_urlopen):
            with patch('os.getenv', return_value='invalid_token'):
                result = _github_get_json("https://api.github.com/test")
                assert result == {"public": "data"}
                assert call_count[0] == 2
    
    def test_api_auth_token_retry_on_403(self):
        """Test auth token retry logic on 403 Forbidden (lines 79-81)."""
        error_403 = HTTPError(
            url="https://api.github.com/test",
            code=403,
            msg="Forbidden",
            hdrs=Message(),
            fp=None
        )
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.read.return_value = b'{"public": "data"}'
        
        call_count = [0]
        
        def mock_urlopen(request, timeout=30):
            call_count[0] += 1
            if call_count[0] == 1:
                raise error_403
            return mock_response
        
        with patch('brmspy.binaries.github.urlopen', side_effect=mock_urlopen):
            with patch('os.getenv', return_value='insufficient_token'):
                result = _github_get_json("https://api.github.com/test")
                assert result == {"public": "data"}
                assert call_count[0] == 2


@pytest.mark.crossplatform
class TestGitHubReleaseAssetMetadata:
    """Test GitHub release asset metadata retrieval."""
    
    def test_asset_not_found_in_release(self):
        """Test error when asset not found in release (line 115)."""
        url = "https://github.com/owner/repo/releases/download/v1.0.0/missing.tar.gz"
        
        mock_release_data = {
            "assets": [
                {"name": "other-asset.tar.gz", "id": 123},
            ]
        }
        
        with patch('brmspy.binaries.github._github_get_json', return_value=mock_release_data):
            with pytest.raises(ValueError, match="Asset 'missing.tar.gz' not found"):
                get_github_release_asset_metadata_from_url(url)
    
    def test_missing_digest_with_require_digest_true(self):
        """Test error when digest missing and require_digest=True (line 126)."""
        url = "https://github.com/owner/repo/releases/download/v1.0.0/asset.tar.gz"
        
        mock_release_data = {
            "assets": [
                {
                    "name": "asset.tar.gz",
                    "id": 123,
                    "browser_download_url": url,
                    "size": 1024,
                    # No "digest" field
                }
            ]
        }
        
        with patch('brmspy.binaries.github._github_get_json', return_value=mock_release_data):
            with pytest.raises(RuntimeError, match="has no digest metadata"):
                get_github_release_asset_metadata_from_url(url, require_digest=True)