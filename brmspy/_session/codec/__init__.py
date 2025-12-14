from __future__ import annotations

"""
Codec registry package (internal).

The session layer serializes arguments/results across the main↔worker boundary via a
`CodecRegistry` populated with a small set of built-in codecs (NumPy, ArviZ, pandas,
dataclasses, and a pickle fallback).

`get_default_registry()` returns the singleton registry instance used by both:

- the main-process proxy (`RModuleSession`)
- the worker loop (`worker_main`)
"""

from .dataclass import *
from .registry import get_default_registry

__all__ = ["get_default_registry"]
