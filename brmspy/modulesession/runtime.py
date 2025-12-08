from __future__ import annotations

from typing import Dict, Any, List


def configure_r_env(runtime_conf: Dict[str, Any]) -> None:
    import os

    r_home = runtime_conf.get("r_home")
    if r_home:
        os.environ["R_HOME"] = r_home

    r_libs_user = runtime_conf.get("r_libs_user")
    if r_libs_user:
        os.environ["R_LIBS_USER"] = r_libs_user


def run_startup_scripts(runtime_conf: Dict[str, Any]) -> None:
    scripts: List[str] = runtime_conf.get("startup", [])
    if not scripts:
        return

    import rpy2.robjects as ro

    for code in scripts:
        ro.r(code)
