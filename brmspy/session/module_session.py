from __future__ import annotations

import atexit
from contextlib import contextmanager
import inspect
import multiprocessing as mp
from pathlib import Path
import subprocess
import uuid
import weakref
from multiprocessing.managers import SharedMemoryManager
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional

from .codec import get_default_registry
from .errors import RSessionError
from .transport import ShmPool, attach_buffers
from .worker import worker_main
import os
import platform


_INTERNAL_ATTRS = {
    "_module",
    "_module_path",
    "_runtime_conf",
    "_mgr",
    "_proc",
    "_conn",
    "_shm_pool",
    "_reg",
    "_closed",
    "_func_cache",
}

def r_home_from_subprocess() -> Optional[str]:
    """Return the R home directory from calling 'R RHOME'."""
    cmd = ('R', 'RHOME')
    tmp = subprocess.check_output(cmd, universal_newlines=True)
    # may raise FileNotFoundError, WindowsError, etc
    r_home = tmp.split(os.linesep)
    if r_home[0].startswith('WARNING'):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    return res

def add_env_defaults(overrides: Dict[str, str]) -> Dict[str, str]:
    """
    Ensure required R environment variables exist inside overrides.
    Mutates overrides in-place and returns it.

    - Ensures R_HOME exists (or auto-detects)
    - Ensures LD_LIBRARY_PATH includes R_HOME/lib (Unix only)
    """
    # ---------------------------------------------------------
    # 1) R_HOME detection / override handling
    # ---------------------------------------------------------
    if "R_HOME" not in overrides:
        r_home = r_home_from_subprocess()
        if not r_home:
            raise RuntimeError(
                "`R_HOME` not provided and cannot auto-detect via subprocess. "
                "R must be installed or explicitly configured."
            )
        r_home_path = Path(r_home)
        overrides["R_HOME"] = r_home_path.as_posix()
    else:
        r_home_path = Path(overrides["R_HOME"])

    # ---------------------------------------------------------
    # 2) LD_LIBRARY_PATH for Unix-like systems
    # ---------------------------------------------------------
    if platform.system() != "Windows":
        r_lib_dir = (r_home_path / "lib").as_posix()

        if "LD_LIBRARY_PATH" not in overrides:
            # Take current LD_LIBRARY_PATH from environment
            current = os.environ.get("LD_LIBRARY_PATH", "")
        else:
            current = overrides["LD_LIBRARY_PATH"]

        # Split into entries (ignore empty ones)
        existing_parts = [p for p in current.split(":") if p]

        # Prepend R_HOME/lib if not already present
        if r_lib_dir not in existing_parts:
            new_ld = ":".join([r_lib_dir] + existing_parts)
        else:
            new_ld = current  # already included

        overrides["LD_LIBRARY_PATH"] = new_ld

    # ---------------------------------------------------------
    return overrides


@contextmanager
def with_env(overrides: Dict[str, str]) -> Iterator[None]:
    """Temporarily apply environment overrides, then restore."""
    overrides = add_env_defaults(overrides)

    old = {}
    sentinel = object()

    for k, v in overrides.items():
        old[k] = os.environ.get(k, sentinel)
        os.environ[k] = v

    try:
        yield
    finally:
        for k, v_old in old.items():
            if v_old is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v_old


def spawn_worker(target, args, env_overrides: Dict[str, str]):
    ctx = mp.get_context("spawn")
    with with_env(env_overrides):
        daemon = os.environ.get("BRMSPY_COVERAGE") != "1" and not os.environ.get("COVERAGE_PROCESS_START")
        proc = ctx.Process(target=target, args=args, daemon=daemon)
        proc.start()
    return proc

class RModuleSession(ModuleType):
    """
    Module-like proxy that forwards function calls to a worker process,
    where the real module is imported and executed.

    All R/rpy2/brms logic lives in that module; this class only does IPC.
    """

    _instances: "weakref.WeakSet[RModuleSession]" = weakref.WeakSet()
    _atexit_registered: bool = False

    def __init__(
        self,
        module: ModuleType,
        module_path: str,
        runtime_conf: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Pretend to be the same module (for IDEs/docs)
        super().__init__(module.__name__, module.__doc__)

        # Store wrapped module and how to import it in worker
        self._module: ModuleType = module
        self._module_path: str = module_path
        self._runtime_conf: Dict[str, Any] = runtime_conf or {}

        # ---- start SHM manager + worker ----
        mgr = SharedMemoryManager()
        mgr.start()

        mgr_address = mgr.address
        mgr_authkey = mgr._authkey  # type: ignore[attr-defined]

        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn

        env_overrides = {
            "BRMSPY_WORKER": "1",
            **(self._runtime_conf.get("env", {})),
        }

        proc = spawn_worker(
            target=worker_main,
            args=(child_conn, mgr_address, mgr_authkey, self._runtime_conf),
            env_overrides=env_overrides
        )

        self._mgr = mgr
        self._proc = proc
        self._shm_pool = ShmPool(mgr)
        self._reg = get_default_registry()
        self._closed = False

        # cache of Python wrappers for functions
        self._func_cache: Dict[str, Callable[..., Any]] = {}

        # copy attributes so IDEs / dir() see the module surface
        self.__dict__.update(module.__dict__)

        # register for global cleanup at exit
        RModuleSession._instances.add(self)
        if not RModuleSession._atexit_registered:
            atexit.register(RModuleSession._cleanup_all)
            RModuleSession._atexit_registered = True

    # ----------------- global cleanup -----------------

    @classmethod
    def _cleanup_all(cls) -> None:
        for inst in list(cls._instances):
            try:
                inst.shutdown()
            except Exception:
                pass

    # ----------------- IPC helpers --------------------

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
                resp.get("error") or "Worker error",
                remote_traceback=resp.get("traceback"),
            )
        pres = resp["result"]
        return self._reg.decode(
            pres["codec"],
            pres["meta"],
            attach_buffers(self._shm_pool, pres["buffers"]),
        )

    def _call_remote(self, func_name: str, *args, **kwargs) -> Any:
        if self._closed:
            raise RuntimeError("RModuleSession is closed")

        target = f"mod:{self._module_path}.{func_name}"
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

    # ----------------- attribute proxying --------------

    def __getattribute__(self, name: str) -> Any:
        # 1. Always allow access to internal attributes via base implementation
        if (
            name in _INTERNAL_ATTRS
            or name.startswith("__") and name.endswith("__")
        ):
            return ModuleType.__getattribute__(self, name)

        # 2. If we already have a cached wrapper for this name, return it
        func_cache = ModuleType.__getattribute__(self, "_func_cache")
        if name in func_cache:
            return func_cache[name]

        module = ModuleType.__getattribute__(self, "_module")

        # 3. If wrapped module has this attribute, decide what to do
        if hasattr(module, name):
            attr = getattr(module, name)

            if callable(attr) and not inspect.isclass(attr):
                # wrap callables so they run in worker
                return self._get_or_create_wrapper(name, attr)

            # non-callables (constants, types, etc.) are just mirrored
            return attr

        # 4. Fallback: use normal module attribute resolution
        return ModuleType.__getattribute__(self, name)

    def _get_or_create_wrapper(self, name: str, orig: Callable[..., Any]):
        func_cache = ModuleType.__getattribute__(self, "_func_cache")
        if name in func_cache:
            return func_cache[name]

        def wrapper(*args, **kwargs):
            return self._call_remote(name, *args, **kwargs)

        wrapper.__name__ = getattr(orig, "__name__", name)
        wrapper.__doc__ = getattr(orig, "__doc__", None)
        wrapper.__wrapped__ = orig  # type: ignore[attr-defined]

        func_cache[name] = wrapper
        return wrapper

    def __dir__(self):
        module = ModuleType.__getattribute__(self, "_module")
        return sorted(set(self.__dict__) | set(dir(module)))

    # ----------------- lifetime ------------------------

    def shutdown(self) -> None:
        if self._closed:
            return

        try:
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
        except Exception:
            pass

        try:
            self._mgr.shutdown()
        except Exception:
            pass

        try:
            self._proc.join(timeout=1.0)
        except Exception:
            pass

        self._closed = True

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
