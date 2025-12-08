from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable


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
    def can_encode(self, obj: Any) -> bool:
        ...

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        ...

    def decode(self, meta: Dict[str, Any],
               buffers: List[memoryview]) -> Any:
        ...


class CodecRegistry:
    def __init__(self) -> None:
        self._by_codec: Dict[str, Encoder] = {}
        self._encoders: List[Encoder] = []

    def register(self, codec_name: str, encoder: Encoder) -> None:
        self._by_codec[codec_name] = encoder
        self._encoders.append(encoder)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        for enc in self._encoders:
            if enc.can_encode(obj):
                res = enc.encode(obj, shm_pool)
                if not res.codec:
                    res.codec = type(enc).__name__
                return res

        # fallback to pickle
        if "pickle" not in self._by_codec:
            raise RuntimeError("No pickle codec registered")
        return self._by_codec["pickle"].encode(obj, shm_pool)

    def decode(self, codec: str, meta: Dict[str, Any],
               buffers: List[memoryview]) -> Any:
        if codec not in self._by_codec:
            raise ValueError(f"Unknown codec: {codec}")
        return self._by_codec[codec].decode(meta, buffers)
