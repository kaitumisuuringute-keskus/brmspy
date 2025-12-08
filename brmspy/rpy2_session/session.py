from __future__ import annotations

import atexit
import multiprocessing as mp
import uuid
import weakref
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Dict

from .codec import get_default_registry
from .errors import RSessionError
from .transport import ShmPool, attach_buffers
from .worker import worker_main


class RSession:
    """
    High-level session API.

    - Spawns a worker process with embedded R/rpy2
    - Uses SharedMemory for large payloads
    """

    _instances: "weakref.WeakSet[RSession]" = weakref.WeakSet()
    _atexit_registered: bool = False

    def __init__(self, runtime_conf: Dict[str, Any] | None = None) -> None:
        self._runtime_conf = runtime_conf or {}

        # SharedMemoryManager server in parent
        self._mgr = SharedMemoryManager()
        self._mgr.start()

        mgr_address = self._mgr.address
        mgr_authkey = self._mgr._authkey  # type: ignore[attr-defined]

        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn

        self._proc = mp.Process(
            target=worker_main,
            args=(child_conn, mgr_address, mgr_authkey, self._runtime_conf),
            daemon=True,
        )
        self._proc.start()

        self._shm_pool = ShmPool(self._mgr)
        self._reg = get_default_registry()
        self._closed = False

        # register for global cleanup at interpreter exit
        RSession._instances.add(self)
        if not RSession._atexit_registered:
            atexit.register(RSession._cleanup_all)
            RSession._atexit_registered = True

    @classmethod
    def _cleanup_all(cls) -> None:
        for sess in list(cls._instances):
            try:
                sess.shutdown()
            except Exception:
                pass

    # ---- internal helpers -------------------------------------------------

    def _encode_arg(self, obj: Any) -> Dict[str, Any]:
        enc = self._reg.encode(obj, self._shm_pool)
        return {
            "codec": enc.codec,
            "meta": enc.meta,
            "buffers": [
                {"name": b.name, "size": b.size}
                for b in enc.buffers
            ],
        }

    def _decode_result(self, resp: Dict[str, Any]) -> Any:
        if not resp["ok"]:
            raise RSessionError(
                resp.get("error") or "Unknown worker error",
                remote_traceback=resp.get("traceback"),
            )
        pres = resp["result"]
        return self._reg.decode(
            pres["codec"],
            pres["meta"],
            attach_buffers(self._shm_pool, pres["buffers"]),
        )

    def _call(self, target: str, *args, **kwargs) -> Any:
        if self._closed:
            raise RuntimeError("RSession is closed")

        req_id = str(uuid.uuid4())
        req = {
            "id": req_id,
            "cmd": "CALL",
            "target": target,
            "args": [self._encode_arg(a) for a in args],
            "kwargs": {k: self._encode_arg(v) for k, v in kwargs.items()},
        }
        self._conn.send(req)
        resp = self._conn.recv()
        return self._decode_result(resp)

    def _eval(self, expr: str) -> Any:
        """
        One-shot evaluation of an R expression, like ro.r("1 + 1").
        """
        if self._closed:
            raise RuntimeError("RSession is closed")

        req_id = str(uuid.uuid4())
        req = {
            "id": req_id,
            "cmd": "EVAL",
            "target": expr,
            "args": [],
            "kwargs": {},
        }
        self._conn.send(req)
        resp = self._conn.recv()
        return self._decode_result(resp)

    # ---- public API -------------------------------------------------------

    def r(self, expr: str) -> Any:
        """
        Evaluate an R expression in the worker R session and return the
        converted Python object (using r_to_py inside the worker).

        Similar to: ro.r(expr)

        Note: if `expr` yields an R function/closure, you probably want
        `session.fun(expr)` instead of relying on this to return a callable.
        """
        return self._eval(expr)

    def fun(self, expr: str):
        """
        Return a callable proxy for an R function expression.

        Examples:
            f = rs.fun("function(a, b) a + b")
            f = rs.fun("dnorm")           # R symbol resolves to function
            f = rs.fun("stats::rnorm")
        """
        return _RFunctionProxy(self, expr)

    def package(self, name: str):
        """
        Return a proxy to an R package:
            brms = rs.package("brms")
            fit = brms.brm(...)
        """
        return _RPackageProxy(self, name)

    def shutdown(self) -> None:
        if self._closed:
            return

        req_id = str(uuid.uuid4())
        self._conn.send(
            {
                "id": req_id,
                "cmd": "SHUTDOWN",
                "target": "",
                "args": [],
                "kwargs": {},
            }
        )
        _ = self._conn.recv()

        self._mgr.shutdown()
        self._proc.join(timeout=1.0)
        self._closed = True

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


class _RFunctionProxy:
    def __init__(self, session: RSession, expr: str) -> None:
        self._session = session
        self._expr = expr

    def __call__(self, *args, **kwargs):
        target = f"r:expr:{self._expr}"
        return self._session._call(target, *args, **kwargs)


class _RPackageProxy:
    def __init__(self, session: RSession, name: str) -> None:
        self._session = session
        self._name = name

    def __getattr__(self, item: str):
        target = f"pkg:{self._name}.{item}"

        def _call(*args, **kwargs):
            return self._session._call(target, *args, **kwargs)

        return _call
