from __future__ import annotations

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
    state_path = get_environments_state_path()
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {"active": env_conf.environment_name}, f, indent=2, ensure_ascii=False
        )
