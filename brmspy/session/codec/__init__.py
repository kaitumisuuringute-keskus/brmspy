from __future__ import annotations

from typing import Optional

from .base import CodecRegistry
from .builtin import InferenceDataCodec, NumpyArrayCodec, PickleCodec
from .dataclass import *
from .registry import get_default_registry

__all__ = ['get_default_registry']
