"""
Tests for brmspy.runtime storage and activation operations.

Focus: Runtime directory validation, hash management, archive installation, activation.

These tests cover runtime storage operations and activation logic.
"""

import pytest
import json
import tarfile
import shutil
from pathlib import Path


@pytest.mark.rdeps
class TestRuntimeDirectoryValidation:
    """Test runtime directory structure validation."""
    
    def test_is_runtime_dir_valid(self, tmp_path):
        """Return True for valid runtime directory"""
        from brmspy.runtime._storage import is_runtime_dir
        
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "manifest.json").write_text("{}")
        
        assert is_runtime_dir(runtime_dir) is True
    
    def test_is_runtime_dir_no_manifest(self, tmp_path):
        """Return False when manifest missing"""
        from brmspy.runtime._storage import is_runtime_dir
        
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        
        assert is_runtime_dir(runtime_dir) is False
    
    def test_is_runtime_dir_not_directory(self, tmp_path):
        """Return False for non-directory"""
        from brmspy.runtime._storage import is_runtime_dir
        
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        assert is_runtime_dir(file_path) is False


@pytest.mark.rdeps
class TestHashManagement:
    """Test hash file operations."""
    
    def test_read_stored_hash_missing(self, tmp_path):
        """Return None when hash file doesn't exist"""
        from brmspy.runtime._storage import read_stored_hash
        
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        
        result = read_stored_hash(runtime_dir)
        assert result is None
    
    def test_read_write_stored_hash(self, tmp_path):
        """Write and read hash file successfully"""
        from brmspy.runtime._storage import read_stored_hash, write_stored_hash
        
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        
        test_hash = "abc123def456789"
        write_stored_hash(runtime_dir, test_hash)
        
        result = read_stored_hash(runtime_dir)
        assert result == test_hash
    
    def test_write_stored_hash_strips_whitespace(self, tmp_path):
        """Strip whitespace from hash when writing"""
        from brmspy.runtime._storage import read_stored_hash, write_stored_hash
        
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        
        test_hash = "  hash_with_spaces  "
        write_stored_hash(runtime_dir, test_hash)
        
        result = read_stored_hash(runtime_dir)
        assert result == "hash_with_spaces"


@pytest.mark.rdeps
class TestInstallFromArchive:
    """Test archive extraction and installation."""
    
    def test_install_from_archive_success(self, tmp_path):
        """Successfully install runtime from archive"""
        from brmspy.runtime._storage import install_from_archive
        
        # Create runtime structure
        stage_dir = tmp_path / "stage"
        runtime_dir = stage_dir / "runtime"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "Rlib").mkdir()
        (runtime_dir / "cmdstan").mkdir()
        
        manifest = {
            "runtime_version": "0.1.0",
            "fingerprint": "test-x86_64-r4.3",
            "r_version": "4.3.0"
        }
        (runtime_dir / "manifest.json").write_text(json.dumps(manifest))
        (runtime_dir / "test_file.txt").write_text("content")
        
        # Create archive
        archive_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(runtime_dir, arcname="runtime")
        
        # Install
        result = install_from_archive(
            archive_path,
            fingerprint="test-x86_64-r4.3",
            version="0.1.0"
        )
        
        # Verify installation
        assert result.exists()
        assert (result / "manifest.json").exists()
        assert (result / "test_file.txt").exists()
        assert (result / "Rlib").is_dir()
        assert (result / "cmdstan").is_dir()
    
    def test_install_from_archive_missing_manifest(self, tmp_path):
        """Raise error when archive has no manifest"""
        from brmspy.runtime._storage import install_from_archive
        
        # Create runtime without manifest
        stage_dir = tmp_path / "stage"
        runtime_dir = stage_dir / "runtime"
        runtime_dir.mkdir(parents=True)
        
        archive_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(runtime_dir, arcname="runtime")
        
        with pytest.raises(RuntimeError, match="Missing manifest.json"):
            install_from_archive(
                archive_path,
                fingerprint="test-fp",
                version="0.1.0"
            )
    
    def test_install_from_archive_replaces_existing(self, tmp_path, monkeypatch):
        """Replace existing runtime when installing"""
        from brmspy.runtime._storage import install_from_archive, get_runtime_base_dir
        
        # Mock base dir to tmp_path
        base_dir = tmp_path / ".brmspy" / "runtime"
        monkeypatch.setattr("brmspy.runtime._storage.get_runtime_base_dir", lambda: base_dir)
        
        # Create existing runtime
        fingerprint = "test-x86_64-r4.3"
        version = "0.1.0"
        existing_runtime = base_dir / f"{fingerprint}-{version}"
        existing_runtime.mkdir(parents=True)
        (existing_runtime / "old_file.txt").write_text("old content")
        
        # Create new archive
        stage_dir = tmp_path / "stage"
        runtime_dir = stage_dir / "runtime"
        runtime_dir.mkdir(parents=True)
        
        manifest = {"runtime_version": version, "fingerprint": fingerprint}
        (runtime_dir / "manifest.json").write_text(json.dumps(manifest))
        (runtime_dir / "new_file.txt").write_text("new content")
        
        archive_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(runtime_dir, arcname="runtime")
        
        # Install over existing
        result = install_from_archive(archive_path, fingerprint, version)
        
        # Old file should be gone, new file present
        assert not (result / "old_file.txt").exists()
        assert (result / "new_file.txt").exists()


@pytest.mark.rdeps
class TestInstallFromDirectory:
    """Test directory-based installation."""
    
    def test_install_from_directory_success(self, tmp_path, monkeypatch):
        """Successfully install runtime from directory"""
        from brmspy.runtime._storage import install_from_directory
        
        # Mock base dir
        base_dir = tmp_path / ".brmspy" / "runtime"
        monkeypatch.setattr("brmspy.runtime._storage.get_runtime_base_dir", lambda: base_dir)
        
        # Create source runtime
        src_dir = tmp_path / "src_runtime"
        src_dir.mkdir()
        manifest = {"runtime_version": "0.1.0", "fingerprint": "test-fp"}
        (src_dir / "manifest.json").write_text(json.dumps(manifest))
        (src_dir / "data.txt").write_text("content")
        
        # Install
        result = install_from_directory(
            src_dir,
            fingerprint="test-fp",
            version="0.1.0"
        )
        
        # Verify
        assert result.exists()
        assert (result / "manifest.json").exists()
        assert (result / "data.txt").exists()
    
    def test_install_from_directory_same_location(self, tmp_path, monkeypatch):
        """Return same path when already in correct location"""
        from brmspy.runtime._storage import install_from_directory, get_runtime_path
        
        # Mock base dir
        base_dir = tmp_path / ".brmspy" / "runtime"
        monkeypatch.setattr("brmspy.runtime._storage.get_runtime_base_dir", lambda: base_dir)
        
        # Create runtime in final location
        fingerprint = "test-fp"
        version = "0.1.0"
        runtime_dir = base_dir / f"{fingerprint}-{version}"
        runtime_dir.mkdir(parents=True)
        manifest = {"runtime_version": version}
        (runtime_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Install to same location
        result = install_from_directory(runtime_dir, fingerprint, version)
        
        assert result == runtime_dir
    
    def test_install_from_directory_missing_manifest(self, tmp_path):
        """Raise error when source has no manifest"""
        from brmspy.runtime._storage import install_from_directory
        
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        with pytest.raises(RuntimeError, match="Missing manifest.json"):
            install_from_directory(src_dir, "test-fp", "0.1.0")


@pytest.mark.rdeps
class TestRuntimeListing:
    """Test runtime listing operations."""
    
    def test_list_installed_runtimes_empty(self, tmp_path, monkeypatch):
        """Return empty list when no runtimes installed"""
        from brmspy.runtime._storage import list_installed_runtimes
        
        base_dir = tmp_path / ".brmspy" / "runtime"
        monkeypatch.setattr("brmspy.runtime._storage.get_runtime_base_dir", lambda: base_dir)
        
        result = list_installed_runtimes()
        assert result == []
    
    def test_list_installed_runtimes_finds_valid(self, tmp_path, monkeypatch):
        """List only valid runtime directories"""
        from brmspy.runtime._storage import list_installed_runtimes
        
        base_dir = tmp_path / ".brmspy" / "runtime"
        base_dir.mkdir(parents=True)
        monkeypatch.setattr("brmspy.runtime._storage.get_runtime_base_dir", lambda: base_dir)
        
        # Create valid runtime
        runtime1 = base_dir / "valid-runtime"
        runtime1.mkdir()
        (runtime1 / "manifest.json").write_text("{}")
        
        # Create invalid directory (no manifest)
        invalid = base_dir / "invalid"
        invalid.mkdir()
        
        # Create file (should be ignored)
        (base_dir / "file.txt").touch()
        
        result = list_installed_runtimes()
        
        assert len(result) == 1
        assert result[0] == runtime1
    
    def test_find_runtime_by_fingerprint(self, tmp_path, monkeypatch):
        """Find runtime matching fingerprint"""
        from brmspy.runtime._storage import find_runtime_by_fingerprint
        
        base_dir = tmp_path / ".brmspy" / "runtime"
        base_dir.mkdir(parents=True)
        monkeypatch.setattr("brmspy.runtime._storage.get_runtime_base_dir", lambda: base_dir)
        
        # Create runtimes with different fingerprints
        fp1 = "linux-x86_64-r4.3"
        runtime1 = base_dir / f"{fp1}-0.1.0"
        runtime1.mkdir()
        (runtime1 / "manifest.json").write_text("{}")
        
        fp2 = "macos-arm64-r4.4"
        runtime2 = base_dir / f"{fp2}-0.2.0"
        runtime2.mkdir()
        (runtime2 / "manifest.json").write_text("{}")
        
        # Find by fingerprint
        result = find_runtime_by_fingerprint(fp1)
        assert result == runtime1
        
        result = find_runtime_by_fingerprint(fp2)
        assert result == runtime2
        
        # Not found
        result = find_runtime_by_fingerprint("nonexistent-fp")
        assert result is None


@pytest.mark.rdeps
class TestRuntimeActivation:
    """Test runtime activation (validation only)."""
    
    def test_activate_validates_structure(self, tmp_path):
        """Activation validates runtime structure"""
        from brmspy.runtime._activation import activate
        
        # Missing manifest
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        
        with pytest.raises(RuntimeError, match="Invalid manifest"):
            activate(runtime_dir)
    
    def test_activate_validates_fingerprint(self, tmp_path):
        """Activation validates system fingerprint"""
        from brmspy.runtime._activation import activate
        from brmspy.runtime._platform import system_fingerprint
        
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "Rlib").mkdir()
        (runtime_dir / "cmdstan").mkdir()
        
        # Wrong fingerprint
        current_fp = system_fingerprint()
        wrong_fp = "wrong-platform-r99.99"
        
        manifest = {
            "runtime_version": "0.1.0",
            "fingerprint": wrong_fp,
            "r_version": "4.3.0"
        }
        (runtime_dir / "manifest.json").write_text(json.dumps(manifest))
        
        if current_fp and current_fp != wrong_fp:
            with pytest.raises(RuntimeError, match="fingerprint mismatch"):
                activate(runtime_dir)