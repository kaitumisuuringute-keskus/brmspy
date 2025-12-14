"""
Environment store helpers (internal).

This module provides small filesystem helpers for brmspy "environments", which
are named directories under `~/.brmspy/environment/<name>/` containing:

- `config.json` (serialized `EnvironmentConfig`)
- `Rlib/` (user-managed R library for that environment)

The session layer uses these helpers when entering/leaving context-managed tools
(e.g. `manage()`), and for convenience methods like environment existence checks.
"""

import json
from pathlib import Path

from brmspy.types.session import EnvironmentConfig


def get_environment_base_dir() -> Path:
    """
    Return the base directory for brmspy environments, creating it if needed.

    Returns
    -------
    pathlib.Path
        `~/.brmspy/environment/`
    """
    base_dir = Path.home() / ".brmspy" / "environment"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_environment_dir(name: str) -> Path:
    """Return the directory for a named environment (may or may not exist)."""
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    return env_dir


def get_environments_state_path() -> Path:
    """Return the path to `environment_state.json` (stores last active environment name)."""
    return Path.home() / ".brmspy" / "environment_state.json"


def get_environment_userlibs_dir(name: str) -> Path:
    """Return the per-environment user library directory: `.../<name>/Rlib`."""
    return get_environment_dir(name=name) / "Rlib"


def get_environment_exists(name: str) -> bool:
    """
    Return True if an environment exists (determined by presence of `config.json`).
    """
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    config_dir = env_dir / "config.json"

    return config_dir.exists()


def get_environment_config(name: str) -> EnvironmentConfig:
    """
    Load an environment configuration from disk.

    Parameters
    ----------
    name : str
        Environment name.

    Returns
    -------
    brmspy.types.session.EnvironmentConfig
        Loaded configuration. If no config file exists, returns a default config
        with `environment_name=name`.
    """
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    config_dir = env_dir / "config.json"

    if not config_dir.exists():
        return EnvironmentConfig(environment_name=name)

    with open(config_dir) as f:
        data = json.load(f)
        return EnvironmentConfig.from_dict(data)
