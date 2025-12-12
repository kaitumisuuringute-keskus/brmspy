import json
from pathlib import Path

from brmspy.types.session_types import EnvironmentConfig


def get_environment_base_dir() -> Path:
    """Returns ~/.brmspy/environment/, creating if needed."""
    base_dir = Path.home() / ".brmspy" / "environment"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_environment_dir(name: str) -> Path:
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    return env_dir


def get_environments_state_path() -> Path:
    return Path.home() / ".brmspy" / "environment_state.json"


def get_environment_userlibs_dir(name: str) -> Path:
    return get_environment_dir(name=name) / "Rlib"


def get_environment_exists(name: str) -> bool:
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    config_dir = env_dir / "config.json"

    return config_dir.exists()


def get_environment_config(name: str) -> EnvironmentConfig:
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    config_dir = env_dir / "config.json"

    if not config_dir.exists():
        return EnvironmentConfig(environment_name=name)

    with open(config_dir) as f:
        data = json.load(f)
        return EnvironmentConfig.from_dict(data)
