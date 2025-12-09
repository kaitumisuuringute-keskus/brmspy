
from typing import Optional

from brmspy.session.codec.base import CodecRegistry
from brmspy.session.codec.builtin import InferenceDataCodec, NumpyArrayCodec, PickleCodec
from brmspy.session.codec.dataclass import register_dataclasses


_default_registry: Optional[CodecRegistry] = None


def get_default_registry() -> CodecRegistry:
    global _default_registry
    if _default_registry is None:
        reg = CodecRegistry()
        reg.register(NumpyArrayCodec())
        reg.register(InferenceDataCodec())

        register_dataclasses(reg)

        # MUST BE LAST
        reg.register(PickleCodec())

        _default_registry = reg
    return _default_registry
