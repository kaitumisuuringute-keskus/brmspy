import json
from pathlib import Path
from typing import Optional

from brmspy.helpers.log import log_warning


def _get_config_path() -> Path:
    """
    Get path to brmspy configuration file.
    
    Returns
    -------
    Path
        Path to ~/.brmspy/config.json
    
    Notes
    -----
    Creates parent directory if it doesn't exist.
    
    Examples
    --------
    ```python
    from brmspy.binaries.config import _get_config_path
    
    config_path = _get_config_path()
    print(config_path)  # ~/.brmspy/config.json
    ```
    """
    config_dir = Path.home() / ".brmspy"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def load_config() -> dict:
    """
    Load brmspy configuration from disk.
    
    Returns
    -------
    dict
        Configuration dictionary, or empty dict if file doesn't exist or is invalid
    
    Notes
    -----
    Silently returns empty dict on any errors (missing file, invalid JSON, etc.).
    This ensures the system continues to work even if config is corrupted.
    
    Examples
    --------
    ```python
    from brmspy.binaries.config import load_config
    
    config = load_config()
    runtime_path = config.get('active_runtime')
    ```
    """
    config_path = _get_config_path()
    
    if not config_path.exists():
        return {}
    
    try:
        with config_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_warning(f"Failed to load config from {config_path}: {e}")
        return {}


def save_config(config: dict) -> None:
    """
    Save brmspy configuration to disk.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    
    Notes
    -----
    Uses atomic write (write to temp file, then rename) to prevent corruption.
    Silently fails if config directory is not writable.
    
    Examples
    --------
    ```python
    from brmspy.binaries.config import save_config
    
    config = {'active_runtime': '/path/to/runtime'}
    save_config(config)
    ```
    """
    config_path = _get_config_path()
    
    try:
        # Atomic write: write to temp file, then rename
        temp_path = config_path.with_suffix('.tmp')
        with temp_path.open('w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Atomic rename (overwrites existing file)
        temp_path.replace(config_path)
    except Exception as e:
        log_warning(f"Failed to save config to {config_path}: {e}")


def get_active_runtime() -> Optional[Path]:
    """
    Get path to currently active prebuilt runtime.
    
    Returns
    -------
    Path or None
        Path to active runtime directory, or None if not configured
    
    Notes
    -----
    Returns None if:
    - No runtime configured in config file
    - Config file doesn't exist
    - Config file is corrupted
    
    Does NOT validate that the path exists - caller should check.
    
    Examples
    --------
    ```python
    from brmspy.binaries.config import get_active_runtime
    
    runtime_path = get_active_runtime()
    if runtime_path and runtime_path.exists():
        print(f"Active runtime: {runtime_path}")
    else:
        print("No active runtime configured")
    ```
    """
    config = load_config()
    runtime_str = config.get('active_runtime')
    
    if runtime_str is None:
        return None
    
    return Path(runtime_str).expanduser().resolve()


def set_active_runtime(runtime_path: Optional[Path]) -> None:
    """
    Save path to active prebuilt runtime.
    
    Parameters
    ----------
    runtime_path : Path
        Path to runtime directory to save as active
    
    Notes
    -----
    Overwrites any existing active runtime configuration.
    Path is normalized (expanded and resolved) before saving.
    
    Examples
    --------
    ```python
    from pathlib import Path
    from brmspy.binaries.config import set_active_runtime
    
    runtime = Path.home() / ".brmspy" / "runtime" / "linux-x86_64-r4.3-0.1.0"
    set_active_runtime(runtime)
    ```
    """
    
    
    config = load_config()
    if runtime_path:
        runtime_path = Path(runtime_path).expanduser().resolve()
        config['active_runtime'] = str(runtime_path)
    else:
        config['active_runtime'] = None
    save_config(config)


def clear_active_runtime() -> None:
    """
    Clear active prebuilt runtime configuration.
    
    Notes
    -----
    Removes the 'active_runtime' key from configuration.
    This will prevent auto-activation on next import.
    
    Examples
    --------
    ```python
    from brmspy.binaries.config import clear_active_runtime
    
    # Disable auto-activation
    clear_active_runtime()
    ```
    """
    config = load_config()
    if 'active_runtime' in config:
        del config['active_runtime']
        save_config(config)