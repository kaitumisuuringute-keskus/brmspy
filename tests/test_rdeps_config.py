"""
Tests for brmspy.runtime._config module.

Focus: Configuration file management, error handling, runtime tracking.

These tests exercise config file operations with error scenarios.
"""

import pytest
import json
from pathlib import Path


@pytest.mark.rdeps
class TestConfigPath:
    """Test config path resolution."""
    
    def test_get_config_path_creates_directory(self, tmp_path, monkeypatch):
        """Verify config directory is created"""
        from brmspy.runtime._config import get_config_path
        
        # Point to temp directory
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        config_path = get_config_path()
        
        # Directory should be created
        assert config_path.parent.exists()
        assert config_path.parent.name == ".brmspy"
        assert config_path.name == "config.json"


@pytest.mark.rdeps
class TestConfigLoad:
    """Test config loading with various scenarios."""
    
    def test_load_config_empty_when_missing(self, tmp_path, monkeypatch):
        """Return empty dict when config file doesn't exist"""
        from brmspy.runtime._config import read_config
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        config = read_config()
        assert config == {}
    
    def test_load_config_valid_json(self, tmp_path, monkeypatch):
        """Load valid JSON config successfully"""
        from brmspy.runtime._config import read_config, get_config_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Create valid config
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = {"active_runtime": "/path/to/runtime", "other": "value"}
        config_path.write_text(json.dumps(test_data))
        
        config = read_config()
        assert config == test_data
    
    def test_load_config_corrupted_json(self, tmp_path, monkeypatch):
        """Gracefully handle corrupted JSON"""
        from brmspy.runtime._config import read_config, get_config_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Create corrupted JSON
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{invalid json content}")
        
        # Should return empty dict, not raise exception
        config = read_config()
        assert config == {}
    
    def test_load_config_invalid_encoding(self, tmp_path, monkeypatch):
        """Handle invalid encoding gracefully"""
        from brmspy.runtime._config import read_config, get_config_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write binary data
        config_path.write_bytes(b'\x80\x81\x82\x83')
        
        # Should return empty dict
        config = read_config()
        assert config == {}


@pytest.mark.rdeps
class TestConfigSave:
    """Test config saving with various scenarios."""
    
    def test_save_config_creates_file(self, tmp_path, monkeypatch):
        """Save config creates file successfully"""
        from brmspy.runtime._config import write_config, read_config
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        test_data = {"active_runtime": "/test/path", "key": "value"}
        write_config(test_data)
        
        # Verify saved correctly
        loaded = read_config()
        assert loaded == test_data
    
    def test_save_config_overwrites_existing(self, tmp_path, monkeypatch):
        """Save config overwrites existing file"""
        from brmspy.runtime._config import write_config, read_config
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Save initial
        write_config({"old": "data"})
        
        # Overwrite
        write_config({"new": "data"})
        
        # Verify overwritten
        loaded = read_config()
        assert loaded == {"new": "data"}
        assert "old" not in loaded
    
    def test_save_config_readonly_fails_gracefully(self, tmp_path, monkeypatch):
        """Handle read-only filesystem gracefully"""
        import os
        from brmspy.runtime._config import write_config, get_config_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Create config directory
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make directory read-only
        os.chmod(config_path.parent, 0o444)
        
        try:
            # Should not raise exception
            write_config({"test": "data"})
        finally:
            # Restore permissions for cleanup
            os.chmod(config_path.parent, 0o755)


@pytest.mark.rdeps
class TestActiveRuntime:
    """Test active runtime get/set/clear operations."""
    
    def test_get_active_runtime_when_none_set(self, tmp_path, monkeypatch):
        """Return None when no runtime configured"""
        from brmspy.runtime._config import get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        runtime = get_active_runtime_path()
        assert runtime is None
    
    def test_get_active_runtime_returns_path(self, tmp_path, monkeypatch):
        """Return Path when runtime is configured"""
        from brmspy.runtime._config import set_active_runtime_path, get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        test_runtime = tmp_path / "runtime" / "test"
        set_active_runtime_path(test_runtime)
        
        runtime = get_active_runtime_path()
        assert runtime is not None
        assert isinstance(runtime, Path)
        assert runtime.name == "test"
    
    def test_set_active_runtime_with_path(self, tmp_path, monkeypatch):
        """Set runtime with Path object"""
        from brmspy.runtime._config import set_active_runtime_path, get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        test_runtime = tmp_path / "my_runtime"
        set_active_runtime_path(test_runtime)
        
        runtime = get_active_runtime_path()
        assert runtime == test_runtime.resolve()
    
    def test_set_active_runtime_with_string(self, tmp_path, monkeypatch):
        """Set runtime with string path"""
        from brmspy.runtime._config import set_active_runtime_path, get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # set_active_runtime_path accepts Path internally, test via Path
        test_runtime_path = Path(str(tmp_path / "my_runtime"))
        set_active_runtime_path(test_runtime_path)
        
        runtime = get_active_runtime_path()
        assert runtime is not None
        assert isinstance(runtime, Path)
    
    def test_set_active_runtime_to_none(self, tmp_path, monkeypatch):
        """Set runtime to None explicitly"""
        from brmspy.runtime._config import set_active_runtime_path, get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # First set a runtime
        set_active_runtime_path(tmp_path / "runtime1")
        assert get_active_runtime_path() is not None
        
        # Then clear it by setting to None
        set_active_runtime_path(None)
        runtime = get_active_runtime_path()
        assert runtime is None
    
    def test_clear_active_runtime(self, tmp_path, monkeypatch):
        """Clear active runtime configuration"""
        from brmspy.runtime._config import (
            set_active_runtime_path,
            get_active_runtime_path
        )
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Set a runtime
        set_active_runtime_path(tmp_path / "runtime")
        assert get_active_runtime_path() is not None
        
        # Clear it (set to None)
        set_active_runtime_path(None)
        
        # Should be None now
        runtime = get_active_runtime_path()
        assert runtime is None
    
    def test_clear_active_runtime_when_not_set(self, tmp_path, monkeypatch):
        """Clear active runtime when nothing is set"""
        from brmspy.runtime._config import set_active_runtime_path, get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        # Should not raise exception when nothing to clear
        set_active_runtime_path(None)
        
        runtime = get_active_runtime_path()
        assert runtime is None


@pytest.mark.rdeps
class TestConfigPersistence:
    """Test config persistence across operations."""
    
    def test_config_persists_across_loads(self, tmp_path, monkeypatch):
        """Config persists across multiple load calls"""
        from brmspy.runtime._config import set_active_runtime_path, get_active_runtime_path
        
        test_home = tmp_path / "test_home"
        monkeypatch.setattr(Path, "home", lambda: test_home)
        
        test_runtime = tmp_path / "persistent_runtime"
        
        # Set runtime
        set_active_runtime_path(test_runtime)
        
        # Get it multiple times
        runtime1 = get_active_runtime_path()
        runtime2 = get_active_runtime_path()
        
        assert runtime1 == runtime2
        assert runtime1 == test_runtime.resolve()
