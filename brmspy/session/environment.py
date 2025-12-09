from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, cast

from brmspy.session.types import EnvironmentConfig


import os

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

def get_environment_config(name: str) -> EnvironmentConfig:
    base_dir = get_environment_base_dir()
    env_dir = base_dir / name
    config_dir = env_dir / "config.json"

    if not config_dir.exists():
        return EnvironmentConfig(environment_name=name)

    with open(config_dir, "r") as f:
        data = json.load(f)
        return EnvironmentConfig.from_dict(data)

def save(env_conf: EnvironmentConfig) -> None:
    base_dir = get_environment_base_dir()
    env_dir = base_dir / env_conf.environment_name
    env_rlib_dir = get_environment_userlibs_dir(name=env_conf.environment_name)
    config_dir = env_dir / "config.json"
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(env_rlib_dir, exist_ok=True)

    with open(config_dir, "w", encoding="utf-8") as f:
        json.dump(env_conf.to_dict(), f, indent=2, ensure_ascii=False)


def save_as_state(env_conf: EnvironmentConfig) -> None:
    state_path = get_environments_state_path()
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({
            "active": env_conf.environment_name
        }, f, indent=2, ensure_ascii=False)


def activate(env_conf: EnvironmentConfig) -> None:
    """only run in worker"""
    from brmspy._runtime import deactivate_runtime, activate_runtime, status, _r_env
    _status = status()

    if env_conf.runtime_path:
        if _status.active_runtime and _status.is_activated:
            if not _status.active_runtime.samefile(env_conf.runtime_path):
                if os.environ.get("BRMSPY_AUTOLOAD") == "1":
                    raise Exception("Can't unload environments when autoload is enabled!")
                deactivate_runtime()
                activate_runtime(env_conf.runtime_path)
    else:
        if _status.active_runtime and _status.is_activated:
            if os.environ.get("BRMSPY_AUTOLOAD") == "1":
                raise Exception("Can't unload environments when autoload is enabled!")
            deactivate_runtime()
    
    userlib_path = get_environment_userlibs_dir(env_conf.environment_name)
    lib_paths = _r_env.get_lib_paths()
    lib_paths = [p for p in lib_paths if ".brmspy/environment/" not in p and ".brmspy\\environment\\" not in p]
    lib_paths.insert(0, userlib_path.as_posix())
    _r_env.set_lib_paths(lib_paths)


    save_as_state(env_conf)



def configure_r_env(env_conf: EnvironmentConfig) -> None:

    r_home = env_conf.r_home
    if r_home:
        os.environ["R_HOME"] = r_home
    
    save(env_conf)
    activate(env_conf)

    



def run_startup_scripts(env_conf: EnvironmentConfig) -> None:
    if not env_conf.startup_scripts:
        return
    scripts: List[str] = env_conf.startup_scripts

    import rpy2.robjects as ro

    for code in scripts:
        ro.r(code)


def _check_r_setup(
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    import shutil, subprocess, os, platform

    ok = True
    messages: List[str] = []

    def info(msg: str) -> None:
        if verbose:
            print(f"[brmspy][INFO] {msg}")

    def warn(msg: str) -> None:
        nonlocal ok
        ok = False
        messages.append(msg)
        if verbose:
            print(f"[brmspy][WARNING] {msg}")

    # --- 1. Try to locate R and RHOME via the R executable -----------------
    r_exec = shutil.which("R")
    r_home_cmd: Optional[str] = None

    if not r_exec:
        # Not necessarily fatal if rpy2 was compiled with an absolute R_HOME,
        # but very suspicious for anything reproducible.
        warn("R executable `R` not found on PATH; this is a fragile setup.")
    else:
        try:
            proc = subprocess.run(
                [r_exec, "RHOME"],
                check=True,
                capture_output=True,
                text=True
            )
            r_home_cmd = proc.stdout.strip()
            if not r_home_cmd:
                warn("`R RHOME` returned an empty value.")
            else:
                info(f"RHOME (from `R RHOME`): {r_home_cmd}")
        except Exception as e:
            warn(f"`R RHOME` failed: {e!r}")

    # --- 2. Look at env vars, but treat them as advisory -------------------
    r_home_env = os.environ.get("R_HOME")
    if r_home_env:
        info(f"R_HOME env: {r_home_env}")
    else:
        info("R_HOME env var not set; relying on rpy2 / system defaults.")

    if platform.system() != "Windows":
        ld = os.environ.get("LD_LIBRARY_PATH")
        if not ld:
            info(
                "LD_LIBRARY_PATH not set; assuming system linker config "
                "already knows where libR.so lives."
            )
        else:
            info(f"LD_LIBRARY_PATH is set (length {len(ld)} chars).")

    # --- 3. Try to import rpy2 and talk to R --------------------------------
    try:
        import rpy2.robjects as ro  # type: ignore[import]
    except Exception as e:
        warn(f"Failed to import rpy2.robjects: {e!r}")
        return ok, messages

    try:
        r_version = str(cast(ro.ListVector, ro.r("R.version"))[0])
        lib_paths = [str(p) for p in cast(ro.ListVector, ro.r(".libPaths()"))]
        info(f"R version: {r_version}")
        info(f".libPaths(): {lib_paths}")
    except Exception as e:
        warn(f"rpy2 could not initialize R / run basic code: {e!r}")
        return ok, messages

    return ok, messages

def _initialise_r_safe() -> None:
    """
    Configure R for safer embedded execution.

    - Try to enforce rpy2 ABI mode (must be set before importing rpy2)
    - Disable fork-based R parallelism (future::multicore, mclapply)
    - Use future::plan(sequential) if future is available
    - Leave cmdstanr multi-core sampling alone
    """

    import os
    import sys

    # CFFI MODE
    if "rpy2" in sys.modules:
        if os.environ.get('RPY2_CFFI_MODE') != "ABI":
            print(
                "[brmspy][WARNING] rpy2 was imported before brmspy; cannot enforce "
                "RPY2_CFFI_MODE (env var). API and BOTH mode are known to cause "
                "instability, ABI is recommended."
            )
    elif os.environ.get('RPY2_CFFI_MODE') in ('BOTH', 'API'):
        print(
            "[brmspy][WARNING] RPY2_CFFI_MODE (env var) is set to API/BOTH. "
            "These modes are known to cause instability and segfaults; "
            "ABI is recommended."
        )
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

    # THREAD SAFETY
    # Could also lead to undefined behaviour if >1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    ok, messages = _check_r_setup()
    if not ok:
        print("[brmspy][WARNING] R environment diagnostics reported problems.")
        for message in messages:
            print(f"[brmspy][WARNING]   {message}")

    import rpy2.robjects as ro

    ro.r(
        r"""
        # Disable fork-based mechanisms that are unsafe in embedded R
        options(
          mc.cores = 1L,             # parallel::mclapply -> serial
          future.fork.enable = FALSE, # disable future::multicore
          loo.cores = 1L # deprecated but still respected, for now.
        )

        # If 'future' is installed, force sequential backend
        if (requireNamespace("future", quietly = TRUE)) {
          future::plan(future::sequential)
        }
        """
    )
