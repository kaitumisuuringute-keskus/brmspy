from __future__ import annotations

from typing import Any, Dict

from multiprocessing.managers import SharedMemoryManager

from brmspy.helpers.conversion import kwargs_r, py_to_r, r_to_py

from .codec import get_default_registry
from .transport import ShmPool, attach_buffers
from .runtime import configure_r_env, run_startup_scripts

def worker_main(conn, mgr_address, mgr_authkey, runtime_conf) -> None:
    """
    Worker entrypoint.

    - Connects to the already-running SharedMemoryManager (started in parent)
    - Sets up R/rpy2
    - Receives CALL/EVAL/SHUTDOWN commands over `conn`
    """

    # Reconnect to the parent's SharedMemoryManager
    smm = SharedMemoryManager(address=mgr_address, authkey=mgr_authkey)
    smm.connect()

    # Configure R env variables before importing rpy2
    configure_r_env(runtime_conf)

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    # Run R-level startup scripts (install packages, set .libPaths, etc.)
    run_startup_scripts(runtime_conf)

    shm_pool = ShmPool(smm)
    reg = get_default_registry()

    pkg_cache: Dict[str, Any] = {}
    expr_cache: Dict[str, Any] = {}

    while True:
        req = conn.recv()
        cmd = req["cmd"]
        req_id = req["id"]

        try:
            if cmd == "SHUTDOWN":
                resp = {
                    "id": req_id,
                    "ok": True,
                    "result": None,
                    "error": None,
                    "traceback": None,
                }
                conn.send(resp)
                break

            elif cmd == "CALL":
                # decode args from shared memory -> Python
                args = [
                    reg.decode(
                        p["codec"],
                        p["meta"],
                        attach_buffers(shm_pool, p["buffers"]),
                    )
                    for p in req["args"]
                ]
                kwargs = {
                    k: reg.decode(
                        p["codec"],
                        p["meta"],
                        attach_buffers(shm_pool, p["buffers"]),
                    )
                    for k, p in req["kwargs"].items()
                }

                # resolve the target (R function)
                target = _resolve_target(
                    req["target"], ro, importr, pkg_cache, expr_cache
                )

                # Python -> R
                args = [py_to_r(o) for o in args]
                kwargs_r_ready = kwargs_r(kwargs)

                # call R
                out_r = target(*args, **kwargs_r_ready)

                # R -> Python
                out = r_to_py(out_r)

                # encode result to shared memory
                enc = reg.encode(out, shm_pool)
                result_payload = {
                    "codec": enc.codec,
                    "meta": enc.meta,
                    "buffers": [
                        {"name": b.name, "size": b.size}
                        for b in enc.buffers
                    ],
                }

                resp = {
                    "id": req_id,
                    "ok": True,
                    "result": result_payload,
                    "error": None,
                    "traceback": None,
                }
                conn.send(resp)

            elif cmd == "EVAL":
                # rs.r(expr): evaluate arbitrary R code once (like ro.r)
                expr = req["target"]

                # no args/kwargs for now; pure expression eval
                out_r = ro.r(expr)
                out = r_to_py(out_r)

                enc = reg.encode(out, shm_pool)
                result_payload = {
                    "codec": enc.codec,
                    "meta": enc.meta,
                    "buffers": [
                        {"name": b.name, "size": b.size}
                        for b in enc.buffers
                    ],
                }

                resp = {
                    "id": req_id,
                    "ok": True,
                    "result": result_payload,
                    "error": None,
                    "traceback": None,
                }
                conn.send(resp)

            else:
                raise ValueError(f"Unknown command: {cmd!r}")

        except Exception as e:  # noqa: BLE001
            import traceback

            tb = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            resp = {
                "id": req_id,
                "ok": False,
                "result": None,
                "error": str(e),
                "traceback": tb,
            }
            conn.send(resp)


def _resolve_target(
    target: str,
    ro,
    importr,
    pkg_cache: Dict[str, Any],
    expr_cache: Dict[str, Any],
):
    if target.startswith("r:expr:"):
        expr = target[len("r:expr:") :]
        fun = expr_cache.get(expr)
        if fun is None:
            fun = ro.r(expr)
            expr_cache[expr] = fun
        return fun

    if target.startswith("pkg:"):
        spec = target[len("pkg:") :]
        if "." not in spec:
            raise ValueError(f"Invalid pkg target (need pkg.func): {target!r}")
        pkg_name, func_name = spec.split(".", 1)

        pkg = pkg_cache.get(pkg_name)
        if pkg is None:
            pkg = importr(pkg_name)
            pkg_cache[pkg_name] = pkg

        if not hasattr(pkg, func_name):
            raise AttributeError(
                f"Package {pkg_name!r} has no function {func_name!r}"
            )
        return getattr(pkg, func_name)

    raise ValueError(f"Unknown target kind: {target!r}")
