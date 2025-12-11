from dataclasses import is_dataclass
from brmspy.helpers.log import log_warning
import brmspy.types as _all_types

from .base import CodecRegistry, DataclassCodec
from .builtin import GenericDataClassCodec

_classes = [
    t
    for name, t in _all_types.__dict__.items()
    if isinstance(t, type) and is_dataclass(t)
]


def register_dataclasses(registry: CodecRegistry):
    from brmspy.types import FitResult

    log_warning(str(_classes))
    # raise Exception()

    for _cls in _classes:
        codec = GenericDataClassCodec(cls=_cls, registry=registry)
        registry.register(codec)

    """fit_result_codec = DataclassCodec(
        cls=FitResult,
        field_codecs={
            "r": "pickle",
            "idata": "arviz.InferenceData",
        },
        registry=registry,
    )
    registry.register(fit_result_codec)"""
