"""
Config file I/O only. No R interaction, no activation logic.
Pure persistence layer.
"""

import json
from pathlib import Path


def get_config_dir() -> Path:
    """Returns ~/.brmspy/, creating if needed."""
    config_dir = Path.home() / ".brmspy"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Returns ~/.brmspy/runtime_state.json."""
    return get_config_dir() / "runtime_state.json"


def read_config() -> dict:
    """Read config file. Returns empty dict if missing/invalid."""
    config_path = get_config_path()

    if not config_path.exists():
        return {}

    try:
        with config_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def write_config(config: dict) -> None:
    """Atomic write to config file (write to temp, rename)."""
    config_path = get_config_path()

    try:
        # Atomic write: write to temp file, then rename
        temp_path = config_path.with_suffix('.tmp')
        with temp_path.open('w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        # Atomic rename (overwrites existing file)
        temp_path.replace(config_path)
    except Exception:
        pass


def get_active_runtime_path() -> Path | None:
    """Read active_runtime from config."""
    config = read_config()
    runtime_str = config.get('active_runtime')

    if runtime_str is None:
        return None

    return Path(runtime_str).expanduser().resolve()


def set_active_runtime_path(path: Path | None) -> None:
    """Write active_runtime to config. Pass None to clear."""
    config = read_config()
    if path:
        path = Path(path).expanduser().resolve()
        config['active_runtime'] = str(path)
    else:
        config['active_runtime'] = None
    write_config(config)
