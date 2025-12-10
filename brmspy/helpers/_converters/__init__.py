import os

assert (
    os.environ.get("BRMSPY_WORKER") == "1"
), "r<->py converters can only be imported in worker mode!"

from ._registry import py_to_r, r_to_py

__all__ = ["py_to_r", "r_to_py"]
