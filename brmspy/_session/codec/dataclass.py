"""
Dataclass codec registration (internal).

This module registers `GenericDataClassCodec` for the public dataclasses that
may cross the main↔worker boundary (primarily result container types and formula
DSL nodes).

The registry is populated at process startup via `get_default_registry()`.
"""

from dataclasses import is_dataclass
from typing import Any

import brmspy.types.brms_results as _all_types
import brmspy.types.formula_dsl as _all_dsl_types

from .base import CodecRegistry
from .builtin import GenericDataClassCodec

_generics: list[type[Any]] = [_all_types.RListVectorExtension]
_classes = [
    t
    for name, t in _all_types.__dict__.items()
    if isinstance(t, type) and is_dataclass(t) and t not in _generics
]
_classes.extend(
    [
        t
        for name, t in _all_dsl_types.__dict__.items()
        if isinstance(t, type) and is_dataclass(t) and t not in _generics
    ]
)
# generics
_classes.extend(_generics)


def register_dataclasses(registry: CodecRegistry) -> None:
    """
    Register codecs for known dataclass types.

    Parameters
    ----------
    registry : brmspy._session.codec.base.CodecRegistry
        Registry to populate.
    """
    for _cls in _classes:
        codec = GenericDataClassCodec(cls=_cls, registry=registry)
        registry.register(codec)
