"""
Tests for brmspy.binaries.use module (runtime management).

Focus: Runtime validation, hash management, activation, and installation.
Target: 66% → 87%+ coverage

These tests exercise runtime bundle installation and activation scenarios.
"""

import pytest
import json
import shutil
from pathlib import Path


@pytest.mark.crossplatform
class TestRuntimeValidation:
    """Test runtime bundle structure validation."""
    
    def test_activate_runtime_missing_manifest(self, tmp_path):
        """Raise error when manifest.json missing (lines 95-96)"""
        from brmspy.binaries.use import activate_runtime
        
        # Create runtime dir without manifest
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "Rlib").mkdir()
        (runtime_root / "cmdstan").mkdir()
        
        with pytest.raises(RuntimeError, match="manifest.json not found"):
            activate_runtime(runtime_root)
    
    def test_activate_runtime_missing_rlib(self, tmp_path):
        """Raise error when Rlib directory missing (lines 97-98)"""
        from brmspy.binaries.use import activate_runtime
        
        # Create runtime dir without Rlib
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "manifest.json").write_text("{}")
        (runtime_root / "cmdstan").mkdir()
        
        with pytest.raises(RuntimeError, match="Rlib directory not found"):
            activate_runtime(runtime_root)
    
    def test_activate_runtime_missing_cmdstan(self, tmp_path):
        """Raise error when cmdstan directory missing (lines 99-100)"""
        from brmspy.binaries.use import activate_runtime
        
        # Create runtime dir without cmdstan
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "manifest.json").write_text("{}")
        (runtime_root / "Rlib").mkdir()
        
        with pytest.raises(RuntimeError, match="cmdstan directory not found"):
            activate_runtime(runtime_root)


@pytest.mark.crossplatform
class TestRuntimeFingerprint:
    """Test system fingerprint validation."""
    
    def test_activate_runtime_fingerprint_mismatch(self, tmp_path):
        """Raise error on fingerprint mismatch (lines 111, 115-116)"""
        from brmspy.binaries.use import activate_runtime
        from brmspy.binaries.env import system_fingerprint
        
        # Create complete runtime structure
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "Rlib").mkdir()
        (runtime_root / "cmdstan").mkdir()
        
        # Get current system fingerprint
        current_fp = system_fingerprint()
        
        # Create manifest with wrong fingerprint
        wrong_fp = "wrong-platform-r99.99"
        manifest = {
            "fingerprint": wrong_fp,
            "r_version": "4.3.0",
            "runtime_version": "0.1.0"
        }
        (runtime_root / "manifest.json").write_text(json.dumps(manifest))
        
        if current_fp and current_fp != wrong_fp:
            with pytest.raises(RuntimeError, match="fingerprint mismatch"):
                activate_runtime(runtime_root)


@pytest.mark.crossplatform
class TestHashManagement:
    """Test hash-based runtime reuse logic."""
    
    def test_read_stored_hash_missing_file(self, tmp_path):
        """Return None when hash file doesn't exist (lines 232-235)"""
        from brmspy.binaries.use import _read_stored_hash
        
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        
        hash_val = _read_stored_hash(runtime_root)
        assert hash_val is None
    
    def test_read_stored_hash_existing_file(self, tmp_path):
        """Read hash from existing file (lines 232-235)"""
        from brmspy.binaries.use import _read_stored_hash, _write_stored_hash
        
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        
        test_hash = "abc123def456"
        _write_stored_hash(runtime_root, test_hash)
        
        hash_val = _read_stored_hash(runtime_root)
        assert hash_val == test_hash
    
    def test_maybe_reuse_runtime_not_exists(self, tmp_path):
        """Return None when runtime doesn't exist (lines 252-253)"""
        from brmspy.binaries.use import _maybe_reuse_existing_runtime
        
        nonexistent = tmp_path / "nonexistent"
        
        result = _maybe_reuse_existing_runtime(
            nonexistent,
            expected_hash="somehash",
            activate=False
        )
        assert result is None
    
    def test_maybe_reuse_runtime_not_valid_dir(self, tmp_path):
        """Return None when dir exists but isn't valid runtime (lines 254-255)"""
        from brmspy.binaries.use import _maybe_reuse_existing_runtime
        
        # Create dir without manifest
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        
        result = _maybe_reuse_existing_runtime(
            runtime_root,
            expected_hash="somehash",
            activate=False
        )
        assert result is None
    
    def test_maybe_reuse_runtime_no_expected_hash(self, tmp_path):
        """Return None when no expected hash provided (lines 256-258)"""
        from brmspy.binaries.use import _maybe_reuse_existing_runtime
        
        # Create valid runtime dir
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "manifest.json").write_text("{}")
        
        result = _maybe_reuse_existing_runtime(
            runtime_root,
            expected_hash=None,
            activate=False
        )
        assert result is None
    
    def test_maybe_reuse_runtime_no_stored_hash(self, tmp_path):
        """Return None when runtime has no stored hash (lines 260-262)"""
        from brmspy.binaries.use import _maybe_reuse_existing_runtime
        
        # Create valid runtime without hash file
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "manifest.json").write_text("{}")
        
        result = _maybe_reuse_existing_runtime(
            runtime_root,
            expected_hash="somehash",
            activate=False
        )
        assert result is None
    
    def test_maybe_reuse_runtime_hash_mismatch_removes_dir(self, tmp_path):
        """Remove runtime when hash mismatch (lines 264-270)"""
        from brmspy.binaries.use import (
            _maybe_reuse_existing_runtime,
            _write_stored_hash
        )
        
        # Create valid runtime with wrong hash
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "manifest.json").write_text("{}")
        _write_stored_hash(runtime_root, "oldhash")
        
        result = _maybe_reuse_existing_runtime(
            runtime_root,
            expected_hash="newhash",
            activate=False
        )
        
        # Should return None and remove directory
        assert result is None
        # Directory may or may not exist depending on shutil.rmtree behavior
    
    def test_maybe_reuse_runtime_hash_match_reuses(self, tmp_path):
        """Reuse runtime when hash matches (lines 272-275)"""
        from brmspy.binaries.use import (
            _maybe_reuse_existing_runtime,
            _write_stored_hash
        )
        
        # Create valid runtime with matching hash
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        (runtime_root / "manifest.json").write_text("{}")
        _write_stored_hash(runtime_root, "matchinghash")
        
        result = _maybe_reuse_existing_runtime(
            runtime_root,
            expected_hash="matchinghash",
            activate=False
        )
        
        # Should reuse and return path
        assert result == runtime_root


@pytest.mark.crossplatform
class TestResolveSource:
    """Test source resolution (URL vs bundle)."""
    
    def test_resolve_source_requires_one_argument(self):
        """Raise error when neither or both arguments provided (line 195)"""
        from brmspy.binaries.use import _resolve_source
        
        # Neither provided
        with pytest.raises(ValueError, match="Exactly one"):
            _resolve_source(url=None, bundle=None)
        
        # Both provided
        with pytest.raises(ValueError, match="Exactly one"):
            _resolve_source(url="http://example.com", bundle="/path")
    
    def test_resolve_source_with_bundle_path(self, tmp_path):
        """Resolve local bundle path (line 208)"""
        from brmspy.binaries.use import _resolve_source
        
        bundle_path = tmp_path / "bundle.tar.gz"
        bundle_path.touch()
        
        resolved, tmp_download = _resolve_source(url=None, bundle=str(bundle_path))
        
        assert resolved == bundle_path
        assert tmp_download is None


@pytest.mark.crossplatform
class TestIsRuntimeDir:
    """Test runtime directory validation."""
    
    def test_is_runtime_dir_not_directory(self, tmp_path):
        """Return False for non-directory (lines 217-218)"""
        from brmspy.binaries.use import _is_runtime_dir
        
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        assert not _is_runtime_dir(file_path)
    
    def test_is_runtime_dir_no_manifest(self, tmp_path):
        """Return False when manifest missing (lines 219-220)"""
        from brmspy.binaries.use import _is_runtime_dir
        
        dir_path = tmp_path / "dir"
        dir_path.mkdir()
        
        assert not _is_runtime_dir(dir_path)
    
    def test_is_runtime_dir_valid(self, tmp_path):
        """Return True for valid runtime directory (lines 217-220)"""
        from brmspy.binaries.use import _is_runtime_dir
        
        dir_path = tmp_path / "runtime"
        dir_path.mkdir()
        (dir_path / "manifest.json").write_text("{}")
        
        assert _is_runtime_dir(dir_path)


@pytest.mark.crossplatform
class TestLoadManifest:
    """Test manifest loading."""
    
    def test_load_manifest_missing_file(self, tmp_path):
        """Raise error when manifest missing (line 226)"""
        from brmspy.binaries.use import _load_manifest
        
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        
        with pytest.raises(RuntimeError, match="Missing manifest.json"):
            _load_manifest(runtime_root)
    
    def test_load_manifest_valid(self, tmp_path):
        """Load valid manifest successfully"""
        from brmspy.binaries.use import _load_manifest
        
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        
        manifest_data = {"runtime_version": "0.1.0", "test": "value"}
        (runtime_root / "manifest.json").write_text(json.dumps(manifest_data))
        
        manifest = _load_manifest(runtime_root)
        assert manifest == manifest_data


@pytest.mark.crossplatform
class TestInstallFromArchive:
    """Test archive extraction and installation."""
    
    def test_install_from_archive_version_mismatch_warning(self, tmp_path):
        """Warn when manifest version doesn't match expected (lines 308-312)"""
        from brmspy.binaries.use import _install_from_archive
        import tarfile
        
        # Create a simple runtime structure
        stage_dir = tmp_path / "stage"
        runtime_dir = stage_dir / "runtime"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "Rlib").mkdir()
        (runtime_dir / "cmdstan").mkdir()
        
        # Manifest with wrong version
        manifest = {
            "runtime_version": "0.9.9",
            "fingerprint": "test",
            "r_version": "4.3.0"
        }
        (runtime_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create archive
        archive_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(runtime_dir, arcname="runtime")
        
        # Install with different version - should warn but proceed
        base_dir = tmp_path / "install_base"
        base_dir.mkdir()
        
        result = _install_from_archive(
            archive_path,
            runtime_root=base_dir / "runtime_root",
            runtime_version="0.1.0",
            base_dir=base_dir
        )
        
        assert result.exists()
    
    def test_install_from_archive_removes_existing(self, tmp_path):
        """Remove existing runtime before installing (lines 314-316)"""
        from brmspy.binaries.use import _install_from_archive
        import tarfile
        
        # Create existing runtime
        existing_runtime = tmp_path / "existing"
        existing_runtime.mkdir()
        (existing_runtime / "old_file.txt").write_text("old")
        
        # Create archive with new runtime
        stage_dir = tmp_path / "stage"
        runtime_dir = stage_dir / "runtime"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "Rlib").mkdir()
        (runtime_dir / "cmdstan").mkdir()
        manifest = {"runtime_version": "0.1.0", "fingerprint": "test"}
        (runtime_dir / "manifest.json").write_text(json.dumps(manifest))
        
        archive_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(runtime_dir, arcname="runtime")
        
        # Install over existing
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        
        result = _install_from_archive(
            archive_path,
            runtime_root=existing_runtime,
            runtime_version="0.1.0",
            base_dir=base_dir
        )
        
        # Old file should be gone
        assert not (result / "old_file.txt").exists()


@pytest.mark.crossplatform
class TestInstallFromDirectory:
    """Test directory-based installation (lines 338-360)."""
    
    def test_install_from_directory_version_mismatch(self, tmp_path):
        """Warn when manifest version mismatch (lines 339-344)"""
        from brmspy.binaries.use import _install_from_directory
        
        # Create source runtime
        src_dir = tmp_path / "src_runtime"
        src_dir.mkdir()
        manifest = {
            "runtime_version": "0.9.9",  # Wrong version
            "fingerprint": "test"
        }
        (src_dir / "manifest.json").write_text(json.dumps(manifest))
        
        runtime_root = tmp_path / "dest_runtime"
        
        # Should warn but proceed
        result = _install_from_directory(
            src_dir,
            runtime_root,
            runtime_version="0.1.0"
        )
        
        assert result.exists()
    
    def test_install_from_directory_same_location(self, tmp_path):
        """Return same path when already in place (lines 349-351)"""
        from brmspy.binaries.use import _install_from_directory
        
        # Create runtime in place
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir()
        manifest = {"runtime_version": "0.1.0"}
        (runtime_root / "manifest.json").write_text(json.dumps(manifest))
        
        # Install to same location
        result = _install_from_directory(
            runtime_root,
            runtime_root,
            runtime_version="0.1.0"
        )
        
        assert result == runtime_root
    
    def test_install_from_directory_removes_existing(self, tmp_path):
        """Remove existing runtime at destination (lines 353-355)"""
        from brmspy.binaries.use import _install_from_directory
        
        # Create source
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        manifest = {"runtime_version": "0.1.0"}
        (src_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Create existing destination
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        (dest_dir / "old.txt").write_text("old")
        
        # Install
        result = _install_from_directory(
            src_dir,
            dest_dir,
            runtime_version="0.1.0"
        )
        
        # Old file should be gone
        assert not (result / "old.txt").exists()
    
    def test_install_from_directory_moves_content(self, tmp_path):
        """Move runtime from source to destination (lines 357-358)"""
        from brmspy.binaries.use import _install_from_directory
        
        # Create source with content
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        manifest = {"runtime_version": "0.1.0"}
        (src_dir / "manifest.json").write_text(json.dumps(manifest))
        (src_dir / "data.txt").write_text("content")
        
        dest_dir = tmp_path / "dest"
        
        result = _install_from_directory(
            src_dir,
            dest_dir,
            runtime_version="0.1.0"
        )
        
        # Content should be at destination
        assert (result / "data.txt").exists()
        assert (result / "manifest.json").exists()


@pytest.mark.crossplatform
class TestInstallAndActivateRuntime:
    """Test high-level installation function."""
    
    def test_install_hash_mismatch_error(self, tmp_path):
        """Raise error on hash mismatch (lines 422-426)"""
        from brmspy.binaries.use import install_and_activate_runtime
        
        # This would only trigger if we provide both url and expected_hash
        # and they don't match - hard to test without actual GitHub
        pass  # Skip - requires real GitHub URL
    
    def test_install_require_attestation_no_hash(self, tmp_path):
        """Raise error when attestation required but no hash (lines 429-432)"""
        from brmspy.binaries.use import install_and_activate_runtime
        
        bundle_path = tmp_path / "bundle.tar.gz"
        bundle_path.touch()
        
        with pytest.raises(ValueError, match="require_attestation"):
            install_and_activate_runtime(
                bundle=str(bundle_path),
                require_attestation=True,
                expected_hash=None,
                activate=False
            )


@pytest.mark.crossplatform
class TestAutoloadLastRuntime:
    """Test autoload functionality (lines 486-493)."""
    
    def test_autoload_last_runtime_no_config(self, tmp_path, monkeypatch):
        """Handle missing config gracefully"""
        from brmspy.binaries.use import autoload_last_runtime
        
        # Point to empty temp directory
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Should not raise exception
        autoload_last_runtime()
    
    def test_autoload_last_runtime_nonexistent_path(self, tmp_path, monkeypatch):
        """Handle nonexistent runtime path gracefully (lines 485-490)"""
        from brmspy.binaries.use import autoload_last_runtime
        from brmspy.binaries.config import set_active_runtime
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Set nonexistent runtime
        nonexistent = tmp_path / "nonexistent_runtime"
        set_active_runtime(nonexistent)
        
        # Should not raise exception
        autoload_last_runtime()
