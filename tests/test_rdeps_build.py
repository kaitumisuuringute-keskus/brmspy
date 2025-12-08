"""
Tests for brmspy.binaries.build module (runtime bundle builder).

Focus: Runtime bundle creation, manifest generation, and archive packing.
Target: 0% → 70%+ coverage

These are integration tests that exercise the actual build process with real
R installations and package metadata.
"""

import pytest
import json
import platform
from pathlib import Path


@pytest.mark.rdeps
class TestBuildManifestHash:
    """Test manifest hash generation for integrity verification."""
    
    def test_generate_manifest_hash_deterministic(self):
        """Verify hash is deterministic for same manifest (lines 58-60)"""
        from brmspy.build._stage import _generate_manifest_hash
        
        manifest = {
            "runtime_version": "0.1.0",
            "fingerprint": "linux-x86_64-r4.3",
            "r_version": "4.3.1",
            "cmdstan_version": "2.33.1",
            "r_packages": {"brms": "2.21.0", "cmdstanr": "0.8.1"}
        }
        
        hash1 = _generate_manifest_hash(manifest)
        hash2 = _generate_manifest_hash(manifest)
        
        # Hash should be deterministic
        assert hash1 == hash2
        # SHA256 produces 64-character hex string
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)
    
    def test_generate_manifest_hash_changes_with_content(self):
        """Verify hash changes when manifest content changes"""
        from brmspy.build._stage import _generate_manifest_hash
        
        manifest1 = {
            "runtime_version": "0.1.0",
            "r_packages": {"brms": "2.21.0"}
        }
        
        manifest2 = {
            "runtime_version": "0.1.0",
            "r_packages": {"brms": "2.21.1"}  # Different version
        }
        
        hash1 = _generate_manifest_hash(manifest1)
        hash2 = _generate_manifest_hash(manifest2)
        
        # Hashes should differ
        assert hash1 != hash2

@pytest.mark.rdeps
class TestBuildRunRJson:
    """Test R JSON execution helper."""
    
    def test_run_r_json_simple(self):
        """Execute simple R code returning JSON (lines 104-107)"""
        from brmspy.build._metadata import _run_r_json
        import rpy2.robjects as ro
        
        # Ensure jsonlite available
        ro.r('if (!requireNamespace("jsonlite", quietly=TRUE)) '
             'install.packages("jsonlite", repos="https://cloud.r-project.org")')
        
        # jsonlite converts scalars to arrays by default, so use auto_unbox=TRUE
        result = _run_r_json('jsonlite::toJSON(list(test="value", num=42), auto_unbox=TRUE)')
        
        assert isinstance(result, dict)
        assert result["test"] == "value"
        assert result["num"] == 42
    
    def test_run_r_json_with_nested_data(self):
        """Test R JSON with nested structures"""
        from brmspy.build._metadata import _run_r_json
        
        result = _run_r_json('''
            jsonlite::toJSON(list(
                r_version = as.character(getRversion()),
                nested = list(a = 1, b = 2)
            ))
        ''')
        
        assert "r_version" in result
        assert "nested" in result
        assert isinstance(result["nested"], dict)


@pytest.mark.rdeps
@pytest.mark.slow
class TestBuildMetadataCollection:
    """Test R environment metadata collection."""

    @classmethod
    def setup_class(cls):
        """Ensure brms and cmdstanr are installed before tests."""
        import brmspy
        brmspy.install_brms(use_prebuilt=True)

    def test_collect_runtime_metadata_structure(self):
        """Collect metadata and verify structure (lines 160-166)"""
        from brmspy.build._metadata import collect_runtime_metadata
        from brmspy.runtime._r_packages import is_package_installed
        
        # Skip if brms or cmdstanr not installed
        if not (is_package_installed("brms") and is_package_installed("cmdstanr")):
            pytest.skip("Requires brms and cmdstanr installed")
        
        metadata = collect_runtime_metadata()
        
        # Verify required keys present
        assert "r_version" in metadata
        assert "cmdstan_path" in metadata
        assert "cmdstan_version" in metadata
        assert "packages" in metadata
        
        # Verify types
        assert isinstance(metadata["r_version"], str)
        assert isinstance(metadata["cmdstan_path"], str)
        assert isinstance(metadata["cmdstan_version"], str)
        assert isinstance(metadata["packages"], list)
    
    def test_collect_runtime_metadata_has_required_packages(self):
        """Verify brms and cmdstanr are included in metadata"""
        from brmspy.build._metadata import collect_runtime_metadata
        from brmspy.runtime._r_packages import is_package_installed
        
        # Skip if brms or cmdstanr not installed
        if not (is_package_installed("brms") and is_package_installed("cmdstanr")):
            pytest.skip("Requires brms and cmdstanr installed")
        
        metadata = collect_runtime_metadata()
        pkg_names = [p["Package"] for p in metadata["packages"]]
        
        # Core packages must be present
        assert "brms" in pkg_names, "brms package not found in metadata"
        assert "cmdstanr" in pkg_names, "cmdstanr package not found in metadata"
        
        # Verify package structure
        for pkg in metadata["packages"]:
            assert "Package" in pkg
            assert "Version" in pkg
            assert "LibPath" in pkg


@pytest.mark.rdeps
@pytest.mark.slow
class TestBuildStaging:
    """Test runtime tree staging and directory structure creation."""

    def test_stage_runtime_tree_creates_structure(self, tmp_path):
        """Verify directory structure creation"""
        from brmspy.build._metadata import collect_runtime_metadata
        from brmspy.build._stage import stage_runtime_tree

        ver = "0.1.0-test1"

        metadata = collect_runtime_metadata()
        runtime_root = stage_runtime_tree(
            tmp_path,
            metadata,
            runtime_version=ver
        )
        
        # Verify basic structure
        assert runtime_root.exists()
        assert (runtime_root / "manifest.json").exists()
        assert (runtime_root / "Rlib").is_dir()
        assert (runtime_root / "cmdstan").is_dir()
        
        # Verify some packages copied to Rlib
        rlib_contents = list((runtime_root / "Rlib").iterdir())
        assert len(rlib_contents) > 0, "No packages in Rlib directory"
        
        # Verify brms and cmdstanr present
        pkg_names = [p.name for p in rlib_contents]
        assert "brms" in pkg_names
        assert "cmdstanr" in pkg_names
    
    def test_stage_runtime_tree_manifest_content(self, tmp_path):
        """Verify manifest.json contains correct metadata"""
        from brmspy.build._metadata import collect_runtime_metadata
        from brmspy.build._stage import stage_runtime_tree
        
        ver = "0.1.0-test2"

        metadata = collect_runtime_metadata()
        runtime_root = stage_runtime_tree(
            tmp_path,
            metadata,
            runtime_version=ver
        )
        
        # Load and verify manifest
        with (runtime_root / "manifest.json").open() as f:
            manifest = json.load(f)
        
        # Verify core fields
        assert manifest["runtime_version"] == ver
        assert "fingerprint" in manifest
        assert "r_version" in manifest
        assert "cmdstan_version" in manifest
        assert "r_packages" in manifest
        assert "manifest_hash" in manifest
        assert "built_at" in manifest
        
        # Verify r_packages structure
        assert isinstance(manifest["r_packages"], dict)
        assert "brms" in manifest["r_packages"]
        assert "cmdstanr" in manifest["r_packages"]
        
        # Verify hash format
        hash_val = manifest["manifest_hash"]
        assert len(hash_val) == 64
        assert all(c in '0123456789abcdef' for c in hash_val)


@pytest.mark.rdeps
@pytest.mark.slow
class TestBuildPacking:
    """Test archive creation from staged runtime."""
    
    def test_pack_runtime_creates_archive(self, tmp_path):
        """Verify archive is created correctly"""
        from brmspy.build._metadata import collect_runtime_metadata
        from brmspy.build._stage import stage_runtime_tree
        from brmspy.build._pack import pack_runtime
        import tarfile
        
        # Stage runtime first
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir()
        ver = "0.1-testb1"
        
        metadata = collect_runtime_metadata()
        runtime_root = stage_runtime_tree(
            stage_dir,
            metadata,
            runtime_version=ver
        )
        
        # Pack it
        out_dir = tmp_path / "out"
        archive_path = pack_runtime(
            runtime_root,
            out_dir,
            runtime_version=ver
        )
        
        # Verify archive exists and has correct name
        assert archive_path.exists()
        assert archive_path.suffix == ".gz"
        assert "brmspy-runtime" in archive_path.name
        assert ver in archive_path.name
        
        # Verify it's a valid tarball
        assert tarfile.is_tarfile(archive_path)
        
        # Verify archive contents
        with tarfile.open(archive_path, "r:gz") as tf:
            names = tf.getnames()
            # Should have runtime/ top-level directory
            assert any(n.startswith("runtime") for n in names)
            # Should contain manifest
            assert any("manifest.json" in n for n in names)
            # Should contain Rlib and cmdstan
            assert any("Rlib" in n for n in names)
            assert any("cmdstan" in n for n in names)
    
    def test_pack_runtime_archive_size(self, tmp_path):
        """Verify packed archive is reasonable size"""
        from brmspy.build._metadata import collect_runtime_metadata
        from brmspy.build._stage import stage_runtime_tree
        from brmspy.build._pack import pack_runtime
        
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir()

        ver = "0.1.0-testb2"
        
        metadata = collect_runtime_metadata()
        runtime_root = stage_runtime_tree(
            stage_dir,
            metadata,
            runtime_version=ver
        )
        
        out_dir = tmp_path / "out"
        archive_path = pack_runtime(
            runtime_root,
            out_dir,
            runtime_version=ver
        )
        
        # Archive should exist and be non-empty
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        assert size_mb > 0.1, f"Archive too small: {size_mb:.2f} MB"
        # Reasonable upper bound (runtime bundles are typically 50-200 MB compressed)
        assert size_mb < 500, f"Archive too large: {size_mb:.2f} MB"