"""
Worker process entrypoint and request loop (internal).

This module runs inside the spawned Python worker process and is responsible for:

- connecting to the parent-owned `SharedMemoryManager`
- initializing a safe embedded-R configuration (see [`_initialise_r_safe()`][brmspy._session.worker.setup._initialise_r_safe])
- decoding incoming IPC requests, executing a resolved target, and encoding results
- converting rpy2 `Sexp` objects to lightweight wrappers via the Sexp cache

The main process should not import rpy2; all embedded-R work happens here.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.managers import SharedMemoryManager
from typing import Any, cast

from rpy2.rinterface_lib.embedded import RRuntimeError

from ...types.session import EnvironmentConfig
from ..codec import get_default_registry
from ..transport import ShmPool, attach_buffers
from .logging import setup_worker_logging
from .setup import _initialise_r_safe, activate, run_startup_scripts
from .sexp_cache import cache_sexp, reattach_sexp

ctx = mp.get_context("spawn")


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

    setup_worker_logging(log_queue)

    import os

    os.environ["BRMSPY_WORKER"] = "1"

    _initialise_r_safe()

    # 1. Connect to SHM manager
    smm = SharedMemoryManager(address=mgr_address, authkey=mgr_authkey, ctx=ctx)
    smm.connect()

    # 2. Optional environment init (R_HOME, R_LIBS_USER, etc.)
    activate(runtime_conf)
    run_startup_scripts(runtime_conf)

    shm_pool = ShmPool(smm)
    reg = get_default_registry()

    module_cache: dict[str, Any] = {}

    import rpy2.rinterface_lib.callbacks
    from rpy2.rinterface_lib.sexp import Sexp

    rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG = (
        "[R]: {exception} {exc_value} {traceback}"
    )

    from ..._singleton._shm_singleton import _set_shm

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

                elif cmd == "PING":
                    conn.send(
                        {
                            "id": req_id,
                            "ok": True,
                            "result": None,
                            "error": None,
                            "traceback": None,
                        }
                    )
                    continue

                elif cmd == "CALL":
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
                    args: list[Any] = reattach_sexp(args)
                    kwargs: dict[str, Any] = reattach_sexp(kwargs)

                    # resolve "mod:pkg.module.func"
                    target = _resolve_module_target(req["target"], module_cache)
                    out = target(*args, **kwargs)
                    out = cache_sexp(out)

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

                elif cmd == "_RUN_TEST_BY_NAME":
                    module = req["kwargs"]["module"]
                    classname = req["kwargs"]["class"]
                    funcname = req["kwargs"]["func"]

                    try:
                        mod = importlib.import_module(module)

                        if classname:
                            cls = getattr(mod, classname)
                            inst = cls()
                            fn = getattr(inst, funcname)
                        else:
                            fn = getattr(mod, funcname)

                        result = fn()

                        enc = reg.encode(result, shm_pool)
                        conn.send(
                            {
                                "id": req_id,
                                "ok": True,
                                "result": {
                                    "codec": enc.codec,
                                    "meta": enc.meta,
                                    "buffers": [
                                        {"name": b.name, "size": b.size}
                                        for b in enc.buffers
                                    ],
                                },
                                "error": None,
                                "traceback": None,
                            }
                        )

                    except Exception as e:
                        import traceback

                        conn.send(
                            {
                                "id": req_id,
                                "ok": False,
                                "result": None,
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                            }
                        )

                else:
                    raise ValueError(f"Unknown command: {cmd!r}")

            except RRuntimeError as e:
                import traceback
                import rpy2.robjects as ro

                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                full_msg = str(e)

                ignore_msgs = ["Can't show last error because no error was recorded"]

                try:

                    # traceback() prints and returns a pairlist -> coerce to something nice
                    r_tb = "\n".join(
                        list(
                            str(v)
                            for v in cast(ro.ListVector, ro.r("unlist(traceback())"))
                        )
                    )
                    tb = r_tb
                except Exception as tb_exc:
                    pass

                # Full base R error message
                try:
                    # full rlang error message (can be multi-line, with bullets etc.)
                    _msg = str(
                        cast(
                            ro.ListVector,
                            ro.r("rlang::format_error_bullets(rlang::last_error())"),
                        )[0]
                    )
                    if _msg and not any(part in _msg for part in ignore_msgs):
                        full_msg = _msg
                    else:
                        raise
                except Exception:
                    # fallback to base R
                    try:
                        _msg = str(cast(ro.ListVector, ro.r("geterrmessage()"))[0])
                        if _msg and not any(part in _msg for part in ignore_msgs):
                            full_msg = _msg
                        else:
                            raise
                    except Exception:
                        pass

                conn.send(
                    {
                        "id": req_id,
                        "ok": False,
                        "result": None,
                        "error": str(full_msg),
                        "traceback": tb,
                    }
                )

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
    Resolve a worker-call target.

    Supported target formats:

    - Module function format:
        mod:pkg.module.func

    - Attribute-chain format (for class-based surfaces, etc):
        mod:pkg.module::Attr.chain.to.callable

      Example:
        mod:brmspy.brms._build_module::BuildModule.collect_runtime_metadata
    """
    if not target.startswith("mod:"):
        raise ValueError(f"Unknown target kind: {target!r}")

    spec = target[len("mod:") :]  # strip "mod:"

    # Module + attribute chain separator
    if "::" in spec:
        mod_name, attr_chain = spec.split("::", 1)
        mod_name = mod_name.strip()
        attr_chain = attr_chain.strip()

        if not mod_name or not attr_chain:
            raise ValueError(f"Invalid module target: {target!r}")

        mod = module_cache.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
            module_cache[mod_name] = mod

        obj: Any = mod
        for part in attr_chain.split("."):
            if not part:
                raise ValueError(f"Invalid module target: {target!r}")
            if not hasattr(obj, part):
                raise AttributeError(
                    f"Target {target!r} missing attribute {part!r} on {obj!r}"
                )
            obj = getattr(obj, part)

        return obj
    else:
        # Module level resolution
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
