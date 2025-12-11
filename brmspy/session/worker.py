# rpy2_session/worker.py

from __future__ import annotations

import importlib
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.managers import SharedMemoryManager
from typing import Any

from ..types.session_types import EnvironmentConfig
from .codec import get_default_registry
from .environment import _initialise_r_safe, configure_r_env, run_startup_scripts
from .transport import ShmPool, attach_buffers
from .worker_logging import setup_worker_logging
from .worker_sexp_cache import cache_sexp, reattach_sexp


def worker_main(
    conn: Connection,
    mgr_address: str | None,
    mgr_authkey: bytes | None,
    runtime_conf: EnvironmentConfig,
    log_queue: mp.Queue,
) -> None:
    """
    Worker entrypoint.

    - Connects to the already-running SharedMemoryManager (started in parent)
    - Optionally configures R env via `runtime_conf`
    - Receives CALL/SHUTDOWN commands over `conn`
    - Executes *Python* functions in modules inside this worker.
      Those modules are free to use rpy2 / brms / cmdstanr however they like.
    """

    import os

    os.environ["BRMSPY_WORKER"] = "1"

    setup_worker_logging(log_queue)

    _initialise_r_safe()

    # 1. Connect to SHM manager
    smm = SharedMemoryManager(address=mgr_address, authkey=mgr_authkey)
    smm.connect()

    # 2. Optional environment init (R_HOME, R_LIBS_USER, etc.)
    configure_r_env(runtime_conf)
    run_startup_scripts(runtime_conf)

    shm_pool = ShmPool(smm)
    reg = get_default_registry()

    module_cache: dict[str, Any] = {}

    import rpy2.rinterface_lib.callbacks
    from rpy2.rinterface_lib.sexp import Sexp

    rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG = (
        "[R]: {exception} {exc_value} {traceback}"
    )

    from .._singleton._shm_singleton import _set_shm

    _set_shm(shm_pool)

    try:
        while True:
            req = conn.recv()
            cmd = req["cmd"]
            req_id = req["id"]

            try:
                if cmd == "SHUTDOWN":
                    conn.send(
                        {
                            "id": req_id,
                            "ok": True,
                            "result": None,
                            "error": None,
                            "traceback": None,
                        }
                    )
                    break

                elif cmd == "CALL":
                    print("REMOTE CALL")
                    # decode Python args
                    args = [
                        reg.decode(
                            p["codec"],
                            p["meta"],
                            attach_buffers(shm_pool, p["buffers"]),
                            p["buffers"],
                            shm_pool=shm_pool,
                        )
                        for p in req["args"]
                    ]
                    kwargs = {
                        k: reg.decode(
                            p["codec"],
                            p["meta"],
                            attach_buffers(shm_pool, p["buffers"]),
                            p["buffers"],
                            shm_pool=shm_pool,
                        )
                        for k, p in req["kwargs"].items()
                    }
                    args = [reattach_sexp(o) for o in args]
                    kwargs = {k: reattach_sexp(v) for k, v in kwargs.items()}

                    # resolve "mod:pkg.module.func"
                    target = _resolve_module_target(req["target"], module_cache)
                    out = target(*args, **kwargs)

                    if isinstance(out, Sexp):
                        out = cache_sexp(out)
                    elif hasattr(out, "r") and isinstance(out.r, Sexp):
                        out.r = cache_sexp(out.r)

                    # encode result
                    enc = reg.encode(out, shm_pool)
                    result_payload = {
                        "codec": enc.codec,
                        "meta": enc.meta,
                        "buffers": [
                            {"name": b.name, "size": b.size} for b in enc.buffers
                        ],
                    }

                    conn.send(
                        {
                            "id": req_id,
                            "ok": True,
                            "result": result_payload,
                            "error": None,
                            "traceback": None,
                        }
                    )

                else:
                    raise ValueError(f"Unknown command: {cmd!r}")

            except Exception as e:
                import traceback

                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                conn.send(
                    {
                        "id": req_id,
                        "ok": False,
                        "result": None,
                        "error": str(e),
                        "traceback": tb,
                    }
                )
    finally:
        pass


def _resolve_module_target(target: str, module_cache: dict[str, Any]):
    """
    Target format: "mod:brmspy.brms.brm"
    """
    if not target.startswith("mod:"):
        raise ValueError(f"Unknown target kind: {target!r}")

    spec = target[len("mod:") :]  # strip "mod:"
    if "." not in spec:
        raise ValueError(f"Invalid module target: {target!r}")

    mod_name, func_name = spec.rsplit(".", 1)

    mod = module_cache.get(mod_name)
    if mod is None:
        mod = importlib.import_module(mod_name)
        module_cache[mod_name] = mod

    if not hasattr(mod, func_name):
        raise AttributeError(f"Module {mod_name!r} has no attribute {func_name!r}")
    return getattr(mod, func_name)
