"""
Codec registry construction helpers (internal).

The session layer uses a single default
[`CodecRegistry`][brmspy._session.codec.base.CodecRegistry] instance per process. The
registry is ordered: the first codec that accepts a value wins, so the registration
order is significant.

Important invariants:
- SHM-backed codecs should be registered before pickle.
- Pickle fallback MUST be registered last.
"""

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
    """
    Return the process-global default codec registry.

    Returns
    -------
    brmspy._session.codec.base.CodecRegistry
        Registry with SHM-first codecs registered, plus a pickle fallback.
    """
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
