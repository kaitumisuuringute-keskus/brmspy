from __future__ import annotations

"""
Parent-side persistence for environment configuration.

This module runs in the main process and is responsible for writing:

- `~/.brmspy/environment/<name>/config.json` (full `EnvironmentConfig`)
- `~/.brmspy/environment_state.json` (last active environment name)

The worker process is restarted for environment changes; the main process persists
the selected configuration on context manager exit.
"""

import json
import os
from pathlib import Path
from typing import cast

from brmspy.types.session import EnvironmentConfig
from .environment import (
    get_environment_base_dir,
    get_environment_userlibs_dir,
    get_environments_state_path,
)


def save(env_conf: EnvironmentConfig) -> None:
    """
    Persist an environment configuration and ensure the directory structure exists.

    Parameters
    ----------
    env_conf : brmspy.types.session.EnvironmentConfig
        Environment configuration to write.
    """
    base_dir = get_environment_base_dir()
    env_dir = base_dir / env_conf.environment_name
    env_rlib_dir = get_environment_userlibs_dir(name=env_conf.environment_name)
    config_dir = env_dir / "config.json"
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(env_rlib_dir, exist_ok=True)

    if "BRMSPY_AUTOLOAD" in env_conf.env:
        del env_conf.env["BRMSPY_AUTOLOAD"]

    with open(config_dir, "w", encoding="utf-8") as f:
        json.dump(env_conf.to_dict(), f, indent=2, ensure_ascii=False)


def save_as_state(env_conf: EnvironmentConfig) -> None:
    """
    Record the active environment name in `environment_state.json`.

    Parameters
    ----------
    env_conf : brmspy.types.session.EnvironmentConfig
        Environment configuration whose name should be recorded.
    """
    state_path = get_environments_state_path()
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {"active": env_conf.environment_name}, f, indent=2, ensure_ascii=False
        )
