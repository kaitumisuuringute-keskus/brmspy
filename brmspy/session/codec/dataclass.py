from dataclasses import is_dataclass
from typing import Any

import brmspy.types.brms_results as _all_types
import brmspy.types.formula_dsl as _all_dsl_types
from brmspy.helpers.log import log_warning

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


def register_dataclasses(registry: CodecRegistry):

    # log_warning(str(_classes))
    # raise Exception()

    for _cls in _classes:
        codec = GenericDataClassCodec(cls=_cls, registry=registry)
        registry.register(codec)
