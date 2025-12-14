from __future__ import annotations
import weakref


"""
Codec registry primitives (internal).

The session layer uses a codec registry to serialize Python values across the
main↔worker boundary. Encoders may allocate shared-memory blocks for large payloads
and return small metadata + SHM references for IPC transport.
"""

from dataclasses import is_dataclass
from typing import Any

from brmspy.types.session import Encoder, EncodeResult
from brmspy.types.shm import ShmBlock, ShmBlockSpec


def _noop(_blocks):
    pass


class CodecRegistry:
    """Ordered registry of encoders used for IPC serialization."""

    def __init__(self) -> None:
        self._by_codec: dict[str, Encoder] = {}
        self._encoders: list[Encoder] = []

    def register(self, encoder: Encoder) -> None:
        """
        Register an encoder instance.

        Parameters
        ----------
        encoder : brmspy.types.session.Encoder
            Encoder to register. Its `codec` attribute is used as the key when present,
            otherwise the class name is used.
        """
        if hasattr(encoder, "codec") and encoder.codec:  # type: ignore
            codec_name = encoder.codec  # type: ignore
        else:
            codec_name = type(encoder).__name__
        self._by_codec[codec_name] = encoder
        encoder.codec = codec_name  # type: ignore
        self._encoders.append(encoder)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        """
        Encode an object by selecting the first encoder that accepts it.

        Parameters
        ----------
        obj : Any
            Value to encode.
        shm_pool : Any
            SHM pool used by codecs for allocating buffers.

        Returns
        -------
        brmspy.types.session.EncodeResult
        """
        for enc in self._encoders:
            if enc.can_encode(obj):
                res = enc.encode(obj, shm_pool)
                if not res.codec:
                    res.codec = type(enc).__name__
                return res

        # fallback to pickle
        if "PickleCodec" not in self._by_codec:
            raise RuntimeError("No pickle codec registered")
        return self._by_codec["PickleCodec"].encode(obj, shm_pool)

    def _attach_shm_lifetime(self, obj: Any, shms: list[ShmBlock]) -> None:
        """Keep SHM blocks alive as long as `obj` is alive."""
        if not shms:
            return
        if obj is None or isinstance(obj, (bool, str, int, float)):
            return

        try:
            weakref.finalize(obj, _noop, tuple(shms))
        except:
            return

    def decode(
        self,
        codec: str,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        """
        Decode a payload using a named codec.

        Parameters
        ----------
        codec : str
            Codec identifier previously returned by `encode()`.
        meta : dict[str, Any]
            Codec metadata.
        buffers : list[memoryview]
            Memoryviews for attached SHM buffers.
        buffer_specs : list[dict]
            Original buffer specs (name/size) corresponding to `buffers`.
        shm_pool : Any
            SHM pool (some codecs may attach additional buffers).

        Returns
        -------
        Any
        """
        if codec not in self._by_codec:
            raise ValueError(
                f"Unknown codec: {codec}, available: {list(self._by_codec.keys())}"
            )
        value = self._by_codec[codec].decode(meta, buffers, buffer_specs, shm_pool)
        self._attach_shm_lifetime(value, buffers)

        return value


class DataclassCodec(Encoder):
    """
    Legacy dataclass codec (internal).

    Prefer `GenericDataClassCodec` in [`brmspy._session.codec.builtin`][brmspy._session.codec.builtin],
    which does not require a per-field codec mapping.

    This class is kept for compatibility and may be removed in the future.
    """

    def __init__(
        self,
        cls: type[Any],
        field_codecs: dict[str, str],  # field_name -> codec_name in registry
        registry: CodecRegistry,
    ) -> None:
        if not is_dataclass(cls):
            raise TypeError(f"{cls!r} is not a dataclass")

        self._cls = cls
        self.codec = cls.__name__
        self._field_codecs = field_codecs
        self._registry = registry

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, self._cls)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        buffers: list[ShmBlockSpec] = []
        fields_meta: dict[str, Any] = {}

        for field_name, codec_name in self._field_codecs.items():
            value = getattr(obj, field_name)

            # Delegate to registry; this will pick the right encoder
            res = self._registry.encode(value, shm_pool)

            start = len(buffers)
            count = len(res.buffers)

            fields_meta[field_name] = {
                "codec": res.codec or codec_name,
                "meta": res.meta,
                "start": start,
                "count": count,
            }

            buffers.extend(res.buffers)

        meta: dict[str, Any] = {
            "cls": self._cls.__qualname__,
            "fields": fields_meta,
        }

        return EncodeResult(
            codec=self.codec,
            meta=meta,
            buffers=buffers,
        )

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        fields_meta: dict[str, Any] = meta["fields"]
        kwargs: dict[str, Any] = {}

        for field_name, fmeta in fields_meta.items():
            codec_name = fmeta["codec"]
            start = fmeta["start"]
            count = fmeta["count"]

            value = self._registry.decode(
                codec_name,
                fmeta["meta"],
                buffers[start : start + count],
                buffer_specs,
                shm_pool,
            )
            kwargs[field_name] = value

        return self._cls(**kwargs)
