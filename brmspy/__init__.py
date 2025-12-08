import os
import sys
import types
from typing import Any
from contextlib import contextmanager


def install_rpy2_stub() -> None:
    # Never stub in the worker
    if os.environ.get("BRMSPY_WORKER") == "1":
        return

    if "rpy2" in sys.modules:
        # Too late; something already imported the real one
        return

    # Top-level stub module
    rpy2_stub = types.ModuleType("rpy2")

    # Submodules for import paths you actually use
    conversion_stub: Any = types.ModuleType("rpy2.robjects.conversion")
    functions_stub: Any = types.ModuleType("rpy2.robjects.functions")
    packages_stub: Any = types.ModuleType("rpy2.robjects.packages")
    vectors_stub: Any = types.ModuleType("rpy2.robjects.vectors")

    # Submodule: rpy2.robjects
    robjects_stub: Any = types.ModuleType("rpy2.robjects")
    # Mark as "package" so submodules under it are importable
    robjects_stub.__path__ = []  # type: ignore[attr-defined]

    # Minimal placeholder types
    class Obj:
        pass

    # Dummy passthrough used for converters / wrappers that should never be
    # called in main; keeps attribute access from exploding.
    def passthrough(*args, **kwargs):
        return args[0] if args else None

    @contextmanager
    def _dummy_localconverter(*args, **kwargs):
        # So `with localconverter(...):` works
        yield None

    # ---- robjects.conversion API ----
    conversion_stub.localconverter = _dummy_localconverter
    robjects_stub.conversion = conversion_stub

    # ---- robjects.functions API ----
    functions_stub.SignatureTranslatedFunction = Obj
    robjects_stub.functions = functions_stub

    # ---- robjects.packages API ----
    def _dummy_importr(*args, **kwargs):
        # If this ever gets called in main, something is wrong – all real
        # package loading must happen in the worker.
        raise RuntimeError(
            "rpy2 stub in main process: `importr` must only be used in worker."
        )

    packages_stub.importr = _dummy_importr
    robjects_stub.packages = packages_stub

    # ---- robjects.vectors API ----
    # You use: `from rpy2.robjects.vectors import StrVector`
    class StrVector(Obj):
        pass

    vectors_stub.StrVector = StrVector
    robjects_stub.vectors = vectors_stub

    # Other things you reference directly on robjects
    robjects_stub.ListVector = Obj
    robjects_stub.default_converter = passthrough
    robjects_stub.passthrough = passthrough
    robjects_stub.pandas2ri = passthrough
    robjects_stub.numpy2ri = passthrough

    # Dummy 'r' – if somebody actually runs R code in main, they get told off
    def _dummy_r(*args, **kwargs):
        raise RuntimeError(
            "rpy2 stub in main process: R code must only run inside worker."
        )

    robjects_stub.r = _dummy_r

    # Attach robjects to top-level rpy2
    rpy2_stub.robjects = robjects_stub

    # ---- rinterface_lib.openrlib.rlock ----
    rinterface_lib_stub = types.ModuleType("rpy2.rinterface_lib")

    class _DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    openrlib_stub = types.SimpleNamespace(rlock=_DummyLock())
    rinterface_lib_stub.openrlib = openrlib_stub

    # Register everything in sys.modules with correct layout
    sys.modules["rpy2"] = rpy2_stub
    sys.modules["rpy2.robjects"] = robjects_stub
    sys.modules["rpy2.robjects.conversion"] = conversion_stub
    sys.modules["rpy2.robjects.functions"] = functions_stub
    sys.modules["rpy2.robjects.packages"] = packages_stub
    sys.modules["rpy2.robjects.vectors"] = vectors_stub
    sys.modules["rpy2.rinterface_lib"] = rinterface_lib_stub
    sys.modules["rpy2.rinterface_lib.openrlib"] = openrlib_stub


install_rpy2_stub()
