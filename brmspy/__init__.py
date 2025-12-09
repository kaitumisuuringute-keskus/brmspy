import os
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Type, cast
from contextlib import contextmanager

from brmspy.session.module_session import RModuleSession

__version__ = "0.2.1"
__author__ = "Remi Sebastian Kits, Adam Haber"
__license__ = "Apache-2.0"

import os
import sys
from types import ModuleType, SimpleNamespace
from contextlib import contextmanager
from typing import Any, Dict


def install_rpy2_stub() -> None:
    # Never stub in the worker
    if os.environ.get("BRMSPY_WORKER") == "1":
        return

    if "rpy2" in sys.modules:
        # Too late; something already imported the real one
        return

    # ----------------- helpers / dummy types -----------------

    class Obj:
        """Minimal placeholder type."""
        def __init__(*args, **kwargs):
            pass

    def passthrough(*args, **kwargs):
        return args[0] if args else None

    @contextmanager
    def _dummy_localconverter(*args, **kwargs):
        # So `with localconverter(...):` works
        yield None

    def _dummy_importr(*args, **kwargs):
        # If this ever gets called in main, something is wrong – all real
        # package loading must happen in the worker.
        raise RuntimeError(
            "rpy2 stub in main process: `importr` must only be used in worker."
        )

    class StrVector(Obj):
        pass

    def _dummy_r(*args, **kwargs):
        raise RuntimeError(
            "rpy2 stub in main process: R code must only run inside worker."
        )

    class _DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    # ----------------- structure description -----------------

    # Each node:
    #   - "package": bool -> whether to set __path__ (so submodules are importable)
    #   - "attrs":   dict -> attributes to set on the module
    #   - "factory": callable(fullname) -> object to register in sys.modules
    #   - "submodules": dict[name -> node]
    STRUCTURE: Dict[str, Dict[str, Any]] = {
        "rpy2": {
            "package": True,
            "attrs": {},
            "submodules": {
                "robjects": {
                    "package": True,
                    "attrs": {
                        # types
                        "ListVector": Obj,
                        "DataFrame": Obj,
                        "Sexp": Obj,
                        # converters / helpers
                        "default_converter": _dummy_localconverter,
                        "passthrough": passthrough,
                        "pandas2ri": passthrough,
                        "numpy2ri": passthrough,
                        # R entrypoint
                        "r": _dummy_r,
                    },
                    "submodules": {
                        "conversion": {
                            "package": False,
                            "attrs": {
                                "localconverter": _dummy_localconverter,
                            },
                            "submodules": {},
                        },
                        "functions": {
                            "package": False,
                            "attrs": {
                                "SignatureTranslatedFunction": Obj,
                            },
                            "submodules": {},
                        },
                        "packages": {
                            "package": False,
                            "attrs": {
                                "importr": _dummy_importr,
                            },
                            "submodules": {},
                        },
                        "vectors": {
                            "package": False,
                            "attrs": {
                                "StrVector": StrVector,
                            },
                            "submodules": {},
                        },
                    },
                },
                # Commonly-present top-level module in rpy2; keep as empty stub
                "rinterface": {
                    "package": True,
                    "attrs": {},
                    "submodules": {},
                },
                "rinterface_lib": {
                    "package": True,
                    "attrs": {},
                    "submodules": {
                        "openrlib": {
                            # openrlib isn't really a module; it's an object.
                            "factory": lambda fullname: SimpleNamespace(
                                rlock=_DummyLock()
                            ),
                            "attrs": {},
                            "submodules": {},
                        },
                        "sexp": {
                            "attrs": {
                                "Sexp": Obj,
                                "NALogicalType": Obj
                            },
                            "submodules": {}
                        }
                    },
                },
            },
        }
    }

    # ----------------- recursive builder -----------------

    def _install_node(
        name: str,
        spec: Dict[str, Any],
        parent_fullname: str | None = None,
    ) -> Any:
        fullname = name if parent_fullname is None else f"{parent_fullname}.{name}"

        # If a factory is provided, treat it as a non-Module object in sys.modules
        factory = spec.get("factory")
        if factory is not None:
            obj = factory(fullname)
            sys.modules[fullname] = obj
            return obj

        # Otherwise, create a normal module
        mod = ModuleType(fullname)

        if spec.get("package"):
            # Mark as "package" so submodules under it are importable
            mod.__path__ = []  # type: ignore[attr-defined]

        # Attach attributes
        for attr_name, attr_value in spec.get("attrs", {}).items():
            setattr(mod, attr_name, attr_value)

        # Register in sys.modules before recursing (for recursive imports)
        sys.modules[fullname] = mod

        # Recurse into submodules
        for sub_name, sub_spec in spec.get("submodules", {}).items():
            child = _install_node(sub_name, sub_spec, fullname)
            setattr(mod, sub_name, child)

        return mod

    # ----------------- build tree -----------------

    # There is only "rpy2" at the root, but keep it generic.
    for root_name, root_spec in STRUCTURE.items():
        _install_node(root_name, root_spec, None)

#install_rpy2_stub()

# -------------------------------------------------------------------
# Typing: describe the brms module surface for static analysis
# -------------------------------------------------------------------
if TYPE_CHECKING:
    # For type checkers / IDE only – can point to the real brms module
    from . import _brms_module
    BrmsModule = _brms_module
else:
    # At runtime, just treat it as a generic module
    BrmsModule = ModuleType  # type: ignore[assignment]

# -------------------------------------------------------------------
# Runtime wiring: proxy in main, real module in worker
# -------------------------------------------------------------------

if os.environ.get("BRMSPY_WORKER") != "1":
    # MAIN PROCESS
    #
    # 1) Ensure rpy2 is stubbed before importing brmspy.brms,
    #    so any top-level rpy2 imports in that module are safe.
    #install_rpy2_stub()

    # 2) Import the heavy brms module; it will see stubbed rpy2 in main.
    from . import _brms_module as _brms_mod

    # 3) Wrap it in RModuleSession so all calls go to the worker.
    _module_path = "brmspy.brms"

    brms = cast(
        BrmsModule,
        RModuleSession(
            module=_brms_mod,
            module_path=_module_path,
            runtime_conf={
                # TODO: pass R_HOME, startup scripts, etc. here if needed
            },
        ),
    )
else:
    # WORKER PROCESS
    #
    # Here we *do not* install the stub – worker must see real rpy2.
    # BRMSPY_WORKER=1 should be set in the worker's env before import.
    from . import _brms_module as brms  # real module

def active():
    return brms


# Export what the package actually provides
__all__ = ['brms']