from brmspy._session.codec.base import CodecRegistry
from brmspy._session.codec.builtin import (
    InferenceDataCodec,
    NumpyArrayCodec,
    PandasDFCodec,
    PickleCodec,
)
from brmspy._session.codec.dataclass import register_dataclasses

_default_registry: CodecRegistry | None = None


def get_default_registry() -> CodecRegistry:
    global _default_registry
    if _default_registry is None:
        reg = CodecRegistry()
        reg.register(NumpyArrayCodec())
        reg.register(InferenceDataCodec())
        reg.register(PandasDFCodec())

        register_dataclasses(reg)

        # MUST BE LAST
        reg.register(PickleCodec())

        _default_registry = reg
    return _default_registry
