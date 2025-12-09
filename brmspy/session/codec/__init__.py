from __future__ import annotations

from typing import Optional

from .base import CodecRegistry
from .builtin import NumpyArrayCodec, PickleCodec, InferenceDataCodec

_default_registry: Optional[CodecRegistry] = None


def get_default_registry() -> CodecRegistry:
    global _default_registry
    if _default_registry is None:
        reg = CodecRegistry()
        reg.register("numpy.ndarray", NumpyArrayCodec())
        reg.register("pickle", PickleCodec())
        reg.register("arviz.InferenceData", InferenceDataCodec())
        
        _default_registry = reg
    return _default_registry
