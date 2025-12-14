from __future__ import annotations

import atexit
import inspect
import logging
import multiprocessing as mp

import os
import platform
import subprocess
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging.handlers import QueueListener
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from types import ModuleType
from typing import Any

from brmspy._session.environment import get_environment_config, get_environment_exists
from brmspy._session.environment_parent import save, save_as_state

from ..types.errors import RSessionError
from ..types.session_types import EnvironmentConfig
from .codec import get_default_registry
from .transport import ShmPool, attach_buffers
from brmspy._session.worker import worker_main

ctx = mp.get_context("spawn")

_INTERNAL_ATTRS = {
    "_module",
    "_module_path",
    "_environment_conf",
    "_mgr",
    "_proc",
    "_conn",
    "_shm_pool",
    "_reg",
    "_closed",
    "_func_cache",
    "_call_remote",
    "_encode_arg",
    "_decode_result",
    "_active_ctx",
    "add_contextmanager",
    "restart",
    "shutdown",
    "environment_exists",
    "environment_activate",
    "_run_test_by_name",
}


def r_home_from_subprocess() -> str | None:
    """Return the R home directory from calling 'R RHOME'."""
    cmd = ("R", "RHOME")
    tmp = subprocess.check_output(cmd, universal_newlines=True)
    # may raise FileNotFoundError, WindowsError, etc
    r_home = tmp.split(os.linesep)
    if r_home[0].startswith("WARNING"):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    return res


def add_env_defaults(overrides: dict[str, str]) -> dict[str, str]:
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

    if "RPY2_CFFI_MODE" not in overrides:
        overrides["RPY2_CFFI_MODE"] = "ABI"

    # ---------------------------------------------------------
    return overrides


@contextmanager
def with_env(overrides: dict[str, str]) -> Iterator[None]:
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


def spawn_worker(
    target, args, env_overrides: dict[str, str], log_queue: mp.Queue | None = None
):
    with with_env(env_overrides):
        daemon = os.environ.get("BRMSPY_TEST") != "1" and not os.environ.get(
            "COVERAGE_PROCESS_START"
        )
        daemon = True
        if log_queue is not None:
            args = (*args, log_queue)
        proc = ctx.Process(target=target, args=args, daemon=daemon)
        proc.start()
    return proc


class ClassProxy:
    """
    Class-like proxy that exposes only @staticmethod members of a surface class
    and executes them in the worker.

    Worker target format:
        mod:{module_path}::{class_name}.{method_name}
    """

    _INTERNAL = {
        "_session",
        "_surface_class",
        "_module_path",
        "_class_name",
        "_allowed",
        "_func_cache",
    }

    def __init__(
        self,
        *,
        session: "RModuleSession",
        surface_class: type,
        module_path: str,
        class_name: str,
    ) -> None:
        self._session = session
        self._surface_class = surface_class
        self._module_path = module_path
        self._class_name = class_name

        # Only expose names backed by `@staticmethod` descriptors.
        allowed: list[str] = []
        for k, v in getattr(surface_class, "__dict__", {}).items():
            if isinstance(v, staticmethod):
                allowed.append(k)
        self._allowed = tuple(sorted(set(allowed)))
        self._func_cache: dict[str, Callable[..., Any]] = {}

    def __getattribute__(self, name: str) -> Any:
        if name in ClassProxy._INTERNAL or (
            name.startswith("__") and name.endswith("__")
        ):
            return object.__getattribute__(self, name)

        allowed = object.__getattribute__(self, "_allowed")
        if name not in allowed:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {name!r}"
            )

        func_cache = object.__getattribute__(self, "_func_cache")
        if name in func_cache:
            return func_cache[name]

        surface_class = object.__getattribute__(self, "_surface_class")
        raw = surface_class.__dict__.get(name)

        # We only allow staticmethod entries; enforce again defensively.
        if not isinstance(raw, staticmethod):
            raise AttributeError(f"{surface_class!r} has no staticmethod {name!r}")

        session = object.__getattribute__(self, "_session")
        module_path = object.__getattribute__(self, "_module_path")
        class_name = object.__getattribute__(self, "_class_name")

        # Grab underlying function only for metadata (__doc__/__name__)
        orig = raw.__func__

        def wrapper(*args, **kwargs):
            return session._call_remote(
                f"mod:{module_path}::{class_name}.{name}", *args, **kwargs
            )

        wrapper.__name__ = getattr(orig, "__name__", name)
        wrapper.__doc__ = getattr(orig, "__doc__", None)
        wrapper.__wrapped__ = orig  # type: ignore[attr-defined]

        func_cache[name] = wrapper
        return wrapper

    def __dir__(self):
        allowed = object.__getattribute__(self, "_allowed")
        return sorted(set(allowed))

    def __repr__(self) -> str:
        module_path = object.__getattribute__(self, "_module_path")
        class_name = object.__getattribute__(self, "_class_name")
        return f"<ClassProxy {module_path}::{class_name}>"

    @property
    def __all__(self) -> list[str]:
        return list(object.__getattribute__(self, "_allowed"))


class RModuleSession(ModuleType):
    """
    Module-like proxy that forwards function calls to a worker process,
    where the real module is imported and executed.

    All R/rpy2/brms logic lives in that module; this class only does IPC.
    """

    _instances: weakref.WeakSet[RModuleSession] = weakref.WeakSet()
    _atexit_registered: bool = False
    _is_rsession: bool = True

    def __init__(
        self,
        module: ModuleType,
        module_path: str,
        environment_conf: EnvironmentConfig | dict[str, Any] | None = None,
    ) -> None:
        # Pretend to be the same module (for IDEs/docs)
        super().__init__(module.__name__, module.__doc__)

        if environment_conf is None:

            try:
                environment_conf = get_environment_config("default")
            except:

                pass

        # Store wrapped module and how to import it in worker
        self._module: ModuleType = module
        self._module_path: str = module_path
        self._environment_conf: EnvironmentConfig = EnvironmentConfig.from_obj(
            environment_conf
        )

        if "BRMSPY_AUTOLOAD" in self._environment_conf.env:
            del self._environment_conf.env["BRMSPY_AUTOLOAD"]

        # cache of Python wrappers for functions
        self._func_cache: dict[str, Callable[..., Any]] = {}

        # Disallow nested tooling contexts (manage/_build/etc)
        self._active_ctx: str | None = None

        # start SHM manager + worker
        self._setup_worker()

        # copy attributes so IDEs / dir() see the module surface
        self.__dict__.update(module.__dict__)

        from .._singleton._shm_singleton import _set_shm

        _set_shm(self._shm_pool)

        # register for global cleanup at exit
        RModuleSession._instances.add(self)
        if not RModuleSession._atexit_registered:
            atexit.register(RModuleSession._cleanup_all)
            RModuleSession._atexit_registered = True

    def _setup_worker(self, autoload=True) -> None:
        """Start SharedMemoryManager and worker process, wire IPC."""

        mgr = SharedMemoryManager(ctx=ctx)
        mgr.start()

        mgr_address = mgr.address
        mgr_authkey = mgr._authkey  # type: ignore[attr-defined]

        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn

        env_overrides: dict[str, str] = {
            "BRMSPY_WORKER": "1",
            **self._environment_conf.env,
        }
        # --- logging bridge: child -> parent ---
        self._log_queue: mp.Queue = ctx.Queue()

        # Use whatever handlers are currently on the root logger.
        root = logging.getLogger()
        self._log_listener = QueueListener(
            self._log_queue,
            *root.handlers,
            respect_handler_level=True,
        )
        self._log_listener.start()

        if autoload:
            env_overrides["BRMSPY_AUTOLOAD"] = "1"
        else:
            env_overrides["BRMSPY_AUTOLOAD"] = "0"

        proc = spawn_worker(
            target=worker_main,
            args=(child_conn, mgr_address, mgr_authkey, self._environment_conf),
            env_overrides=env_overrides,
            log_queue=self._log_queue,
        )

        self._mgr = mgr
        self._proc = proc
        self._shm_pool = ShmPool(mgr)
        self._reg = get_default_registry()
        self._closed = False

        # --- handshake: wait until worker is ready ---
        # This is MANDATORY. Unless we want zombies or race conditions.
        req_id = str(uuid.uuid4())
        self._conn.send(
            {
                "id": req_id,
                "cmd": "PING",
                "target": "",
                "args": [],
                "kwargs": {},
            }
        )
        if not self._conn.poll(10.0):
            # worker never replied -> treat as startup failure,
            # clean up and raise
            self._teardown_worker()
            raise RuntimeError("Worker failed to start within timeout")

        resp = self._conn.recv()
        if not resp.get("ok", False):
            self._teardown_worker()
            raise RuntimeError(f"Worker failed to initialize: {resp.get('error')}")

    def _teardown_worker(self) -> None:
        """Internal helper to stop worker/manager; used by shutdown/restart."""
        if self._closed:
            return

        # best-effort graceful shutdown
        try:
            if self._conn:
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
                # wait for ack, but don't block forever
                if self._conn.poll(5.0):
                    _ = self._conn.recv()
        except Exception:
            pass

        # stop logging listener
        try:
            listener = getattr(self, "_log_listener", None)
            if listener is not None:
                listener.stop()
        except Exception:
            pass

        # shut down SHM manager
        try:
            self._mgr.shutdown()
        except Exception:
            pass

        # give worker a chance to exit, then kill it if needed
        # This is MANDATORY. Unless we want zombies or race conditions running amock
        try:
            if self._proc is not None:
                self._proc.join(timeout=5.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=5.0)
        except Exception:
            pass

        self._closed = True

    # ----------------- global cleanup -----------------

    @classmethod
    def _cleanup_all(cls) -> None:
        for inst in list(cls._instances):
            try:
                inst.shutdown()
            except Exception:
                pass

    # ----------------- IPC helpers --------------------

    def _encode_arg(self, obj: Any) -> dict[str, Any]:
        enc = self._reg.encode(obj, self._shm_pool)
        return {
            "codec": enc.codec,
            "meta": enc.meta,
            "buffers": [{"name": b.name, "size": b.size} for b in enc.buffers],
        }

    def _decode_result(self, resp: dict[str, Any]) -> Any:
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
            pres["buffers"],
            shm_pool=self._shm_pool,
        )

    def _call_remote(self, func_name: str, *args, **kwargs) -> Any:
        if self._closed:
            raise RuntimeError("RModuleSession is closed")

        if func_name.startswith("mod:"):
            target = func_name
        else:
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
        if name in _INTERNAL_ATTRS or name.startswith("__") and name.endswith("__"):
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
        self._teardown_worker()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def add_contextmanager(
        self,
        *,
        surface_class: type,
        surface_class_path: str,
    ):
        """
        Attach a surface (module or class) as a context manager factory.

            Yields a ClassProxy exposing only @staticmethod members of the class.
            Callables execute in worker via:
                mod:{module_path}::{ClassName}.{name}
            where `module_path` and `ClassName` are derived from `surface_class_path`.

        Semantics match current `brmspy._session.manage.manage()`:
        - On enter: resolve env config and restart worker with autoload=False
        - On exit: persist env via save()/save_as_state(); do not restore old config
        - Nesting is not allowed (raises)
        """

        session = self

        if surface_class is None or surface_class_path is None:
            raise ValueError("surface_class and surface_class_path must both be set.")
        if "." not in surface_class_path:
            raise ValueError(
                "surface_class_path must look like 'pkg.module.ClassName'."
            )

        module_path, class_name = surface_class_path.rsplit(".", 1)
        ctx_label = f"{module_path}::{class_name}"

        class _Ctx:
            def __init__(
                self,
                *,
                environment_config: EnvironmentConfig | dict[str, str] | None = None,
                environment_name: str | None = None,
            ) -> None:
                self._environment_config = environment_config
                self._environment_name = environment_name
                self._new_conf: EnvironmentConfig | None = None

            def __enter__(self):
                if session._active_ctx is not None:
                    raise RuntimeError(
                        f"Nested brmspy contexts are not supported "
                        f"(active={session._active_ctx!r}, new={ctx_label!r})."
                    )
                session._active_ctx = ctx_label

                if self._environment_name and self._environment_config:
                    session._active_ctx = None
                    raise Exception(
                        "Only provide one: environment name or environment config"
                    )

                if not self._environment_name and self._environment_config:
                    overrides = EnvironmentConfig.from_obj(self._environment_config)
                elif self._environment_name:
                    overrides = get_environment_config(self._environment_name)
                else:
                    overrides = None

                old_conf = session._environment_conf
                new_conf = overrides if overrides else old_conf
                self._new_conf = new_conf

                # fresh worker with new_conf
                session.restart(environment_conf=new_conf, autoload=False)

                return ClassProxy(
                    session=session,
                    surface_class=surface_class,
                    module_path=module_path,
                    class_name=class_name,
                )

            def __exit__(self, exc_type, exc, tb) -> None:
                try:
                    if self._new_conf is not None:
                        save(self._new_conf)
                        save_as_state(self._new_conf)
                finally:
                    session._active_ctx = None
                return None

        def factory(
            *,
            environment_config: EnvironmentConfig | dict[str, str] | None = None,
            environment_name: str | None = None,
        ):
            return _Ctx(
                environment_config=environment_config, environment_name=environment_name
            )

        return factory

    def restart(
        self,
        environment_conf: dict[str, Any] | EnvironmentConfig | None = None,
        autoload: bool = True,
    ) -> None:
        """
        Restart the underlying worker process and shared-memory manager.

        If `environment_conf` is provided, it replaces the existing configuration
        for the new worker; otherwise the existing `self._environment_conf` is reused.
        """
        if environment_conf is not None:
            self._environment_conf = EnvironmentConfig.from_obj(environment_conf)

        # Tear down existing worker (if any)
        self._teardown_worker()

        # Optional: clear wrappers if you want a fully "fresh" view.
        # They are safe to reuse, but clearing them forces re-resolution.
        self._func_cache.clear()

        # Start a fresh worker with current env conf
        self._setup_worker(autoload=autoload)

    def environment_exists(self, name: str):
        return get_environment_exists(name)

    def environment_activate(self, name: str):
        manage = self.manage
        if manage:
            with manage(environment_name=name) as ctx:
                pass
        else:
            raise Exception("Invalid state. manage is not defined!")

    def _run_test_by_name(
        self, module_path: str, class_name: str | None, func_name: str
    ):
        """Run a test identified by module/class/function INSIDE the worker."""
        if os.getenv("BRMSPY_TEST") != "1":
            raise RuntimeError("BRMSPY_TEST=1 required for worker test execution")

        req_id = str(uuid.uuid4())
        self._conn.send(
            {
                "id": req_id,
                "cmd": "_RUN_TEST_BY_NAME",
                "target": "",
                "args": [],
                "kwargs": {
                    "module": module_path,
                    "class": class_name,
                    "func": func_name,
                },
            }
        )

        resp = self._conn.recv()

        if not resp.get("ok", False):
            raise RSessionError(
                resp.get("error", "Worker test failed"),
                remote_traceback=resp.get("traceback"),
            )

        pres = resp["result"]

        return self._reg.decode(
            pres["codec"],
            pres["meta"],
            attach_buffers(self._shm_pool, pres["buffers"]),
            pres["buffers"],
            shm_pool=self._shm_pool,
        )
