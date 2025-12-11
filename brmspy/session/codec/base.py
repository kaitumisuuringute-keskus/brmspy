from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, List, Protocol, Type, runtime_checkable


@dataclass
class ShmBlockSpec:
    name: str
    size: int


@dataclass
class EncodeResult:
    codec: str
    meta: Dict[str, Any]
    buffers: List[ShmBlockSpec]


@runtime_checkable
class Encoder(Protocol):
    def can_encode(self, obj: Any) -> bool: ...

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult: ...

    def decode(
        self,
        meta: Dict[str, Any],
        buffers: List[memoryview],
        buffer_specs: List[dict],
        shm_pool: Any,
    ) -> Any: ...


class CodecRegistry:
    def __init__(self) -> None:
        self._by_codec: Dict[str, Encoder] = {}
        self._encoders: List[Encoder] = []

    def register(self, encoder: Encoder) -> None:
        if hasattr(encoder, "codec") and encoder.codec:  # type: ignore
            codec_name = encoder.codec  # type: ignore
        else:
            codec_name = type(encoder).__name__
        self._by_codec[codec_name] = encoder
        encoder.codec = codec_name  # type: ignore
        self._encoders.append(encoder)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
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

    def decode(
        self,
        codec: str,
        meta: Dict[str, Any],
        buffers: List[memoryview],
        buffer_specs: List[Dict],
        shm_pool: Any,
    ) -> Any:
        if codec not in self._by_codec:
            raise ValueError(
                f"Unknown codec: {codec}, available: {list(self._by_codec.keys())}"
            )
        return self._by_codec[codec].decode(meta, buffers, buffer_specs, shm_pool)


class DataclassCodec(Encoder):
    """
    Generic codec that encodes/decodes dataclasses by delegating
    each field to a registered codec in CodecRegistry.
    """

    def __init__(
        self,
        cls: Type[Any],
        field_codecs: Dict[str, str],  # field_name -> codec_name in registry
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
        buffers: List[ShmBlockSpec] = []
        fields_meta: Dict[str, Any] = {}

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

        meta: Dict[str, Any] = {
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
        meta: Dict[str, Any],
        buffers: List[memoryview],
        buffer_specs: List[dict],
        shm_pool: Any,
    ) -> Any:
        fields_meta: Dict[str, Any] = meta["fields"]
        kwargs: Dict[str, Any] = {}

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
